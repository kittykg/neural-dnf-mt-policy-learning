from collections import OrderedDict
import random
from pathlib import Path
import time
import traceback
from typing import Any


import gymnasium as gym
from gymnasium.envs.toy_text.taxi import TaxiEnv
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter  # type: ignore
import wandb

from neural_dnf import NeuralDNF, NeuralDNFEO, NeuralDNFMutexTanh
from neural_dnf.neural_dnf import BaseNeuralDNF  # for type hinting
from neural_dnf.utils import DeltaDelayedExponentialDecayScheduler

from common import init_params
from taxi_common import *
from utils import post_to_discord_webhook

PPO_MODEL_DIR = Path(__file__).parent / "taxi_ppo_storage/"
if not PPO_MODEL_DIR.exists() or not PPO_MODEL_DIR.is_dir():
    PPO_MODEL_DIR.mkdir(parents=True, exist_ok=True)


class TaxiEnvPPOBaseAgent(nn.Module):
    """
    To create a base agent, pass in the following parameters:
    - num_latent: The latent size of the agent
    - use_decode_obs: Whether to use the decoded observation or not
    """

    # Model components
    actor: nn.Module
    critic: nn.Module

    # Actor parameters
    num_inputs: int
    num_latent: int
    action_size: int = N_ACTIONS

    # Flag to use decode observation
    use_decode_obs: bool
    input_key: str

    def __init__(self, num_latent: int, use_decode_obs: bool):
        super().__init__()
        self.num_latent = num_latent
        self.use_decode_obs = use_decode_obs

        self.num_inputs = (
            N_DECODE_OBSERVATION_SIZE
            if self.use_decode_obs
            else N_OBSERVATION_SIZE
        )

        if self.use_decode_obs:
            self.input_key = "decode_input"
        else:
            self.input_key = "input"

        self.actor = self._create_default_actor()
        self.critic = self._create_default_critic()

        self._init_params()

    def get_value(self, preprocessed_obs: dict[str, Tensor]) -> Tensor:
        """
        Return the value of the state.
        This function is used in PPO algorithm
        """
        return self.critic(preprocessed_obs[self.input_key])

    def get_action_and_value(
        self, preprocessed_obs: dict[str, Tensor], action=None
    ) -> tuple[Tensor, Any, Any, Tensor]:
        """
        Return the action, log probability of the action, entropy of the action
        distribution, and the value of the state.
        This function is used in PPO algorithm
        """
        x = preprocessed_obs[self.input_key]
        logits = self.actor(x)
        dist = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        return (
            action,
            dist.log_prob(action),
            dist.entropy(),
            self.critic(x),
        )

    def get_actions(
        self, preprocessed_obs: dict[str, Tensor], use_argmax: bool = True
    ):
        """
        Return the actions based on the observation.
        """
        x = preprocessed_obs[self.input_key]
        logits = self.actor(x)
        dist = torch.distributions.Categorical(logits=logits)

        actions = dist.probs.max(dim=1)[1] if use_argmax else dist.sample()  # type: ignore
        return actions.cpu().numpy()

    def get_action_distribution(
        self, preprocessed_obs: dict[str, Tensor]
    ) -> Categorical:
        """
        Return the action distribution based on the observation.
        """
        x = preprocessed_obs[self.input_key]
        logits = self.actor(x)
        return torch.distributions.Categorical(logits=logits)

    def _create_default_actor(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.num_inputs, self.num_latent),
            nn.Tanh(),
            nn.Linear(self.num_latent, self.action_size),
        )

    def _create_default_critic(self) -> nn.Module:
        if self.use_decode_obs:
            return nn.Sequential(
                nn.Linear(self.num_inputs, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
        else:
            return nn.Sequential(
                nn.Linear(self.num_inputs, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
            )

    def _init_params(self) -> None:
        self.apply(init_params)


class TaxiEnvPPOMLPAgent(TaxiEnvPPOBaseAgent):
    """
    An agent for gymnasium Taxi environment, with a 2-layer MLP actor.
    To create a `TaxiEnvPPOMLPAgent` agent, pass in the following parameters:
    - num_latent (int): the number of latent features
    - use_decode_obs (bool): flag to use decode observation

    The actor and critic networks are created using `_create_default_actor()`
    and `_create_default_critic()` methods respectively.
    """


class TaxiEnvPPONDNFBasedAgent(TaxiEnvPPOBaseAgent):
    """
    Base class for agents using a neural DNF module as the actor.
    """

    actor: BaseNeuralDNF

    def _create_default_actor(self) -> BaseNeuralDNF:
        # This method should be overridden by the subclass
        raise NotImplementedError

    def get_aux_loss(
        self, preprocessed_obs: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """
        Return the auxiliary loss dictionary for the agent.
        The keys are:
        - l_disj_l1_mod: disjunction weight regularisation loss
        - l_tanh_conj: tanh conjunction output regularisation loss
        """
        # Disjunction weight regularisation loss
        p_t = torch.cat(
            [p.view(-1) for p in self.actor.disjunctions.parameters()]
        )
        l_disj_l1_mod = torch.abs(p_t * (6 - torch.abs(p_t))).mean()

        # Push tanhed conjunction output towards -1 and 1 only
        x = preprocessed_obs[self.input_key]
        tanh_conj = torch.tanh(self.actor.conjunctions(x))
        l_tanh_conj = (1 - tanh_conj.abs()).mean()

        return {
            "l_disj_l1_mod": l_disj_l1_mod,
            "l_tanh_conj": l_tanh_conj,
        }

    def get_actor_output(
        self,
        preprocessed_obs: dict[str, Tensor],
    ) -> Tensor:
        """
        Return the raw output of the actor (before tanh)
        This function should only be called during evaluation.
        """
        assert (
            not self.training
        ), "get_actor_output() should only be called during evaluation!"

        with torch.no_grad():
            x = preprocessed_obs[self.input_key]
            return self.actor(x)

    def get_actions(
        self,
        preprocessed_obs: dict[str, Tensor],
        use_argmax: bool = True,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        """
        This function should only be called during evaluation.
        Because of the use of neural DNF module, the output of the actor can be
        treated as a symbolic output after tanh. This function returns both the
        probabilistic/argmax based action and the tanh action.
        """
        assert (
            not self.training
        ), "get_actions() should only be called during evaluation!"

        with torch.no_grad():
            raw_actions = self.get_actor_output(preprocessed_obs)
        dist = torch.distributions.Categorical(logits=raw_actions)
        if use_argmax:
            actions = dist.probs.max(1)[1]  # type: ignore
        else:
            actions = dist.sample()
        tanh_action = torch.tanh(raw_actions)

        return actions.cpu().numpy(), tanh_action.cpu().numpy()

    def load_critic_from_trained_model(
        self, model_path: Path, disable_critic_training: bool = False
    ):
        """
        Load the critic from a trained model.
        """
        full_model = torch.load(model_path)
        critic_dict = OrderedDict()
        for k in full_model.keys():
            if "critic" in k:
                ck = k.replace("critic.", "")
                critic_dict[ck] = full_model[k]
        self.critic.load_state_dict(critic_dict)

        if disable_critic_training:
            for param in self.critic.parameters():
                param.requires_grad = False


class TaxiEnvPPONDNFAgent(TaxiEnvPPONDNFBasedAgent):
    """
    An agent for gymnasium Taxi environment, with `NeuralDNF` as actor.
    This agent is not usually expected to use for training. This agent is more
    expected to be used as a post-training evaluation agent from either a
    trained `TaxiEnvPPONDNFEOAgent` or `TaxiEnvPPONDNFMutexTanhAgent`.
    To create a `TaxiEnvPPONDNFAgent` agent, pass in the following parameters:
    - num_latent (int): the number of conjunctions allowed in NDNF
    - use_decode_obs (bool): flag to use decode observation
    """

    actor: NeuralDNF

    def _create_default_actor(self) -> NeuralDNF:
        return NeuralDNF(
            self.num_inputs, self.num_latent, self.action_size, 1.0
        )


class TaxiEnvPPONDNFEOAgent(TaxiEnvPPOBaseAgent):
    """
    An agent for gymnasium Taxi environment, with `NeuralDNFEO` actor.
    This agent is used for training, and to be converted to a
    `TaxiEnvPPONDNFAgent` for post-training evaluation.
    To create a `TaxiEnvPPONDNFMTAgent` agent, pass in the following parameters:
    - num_latent (int): the number of conjunctions allowed in NDNF-EO
    - use_decode_obs (bool): flag to use decode observation
    """

    actor: NeuralDNFEO

    def _create_default_actor(self) -> NeuralDNFEO:
        return NeuralDNFEO(
            self.num_inputs, self.num_latent, self.action_size, 1.0
        )

    def to_ndnf_agent(self) -> TaxiEnvPPONDNFAgent:
        """
        Convert this agent to a TaxiEnvPPONDNFAgent.
        """
        ndnf_agent = TaxiEnvPPONDNFAgent(self.num_latent, self.use_decode_obs)
        ndnf_agent.actor = self.actor.to_ndnf()
        return ndnf_agent


class TaxiEnvPPONDNFMTAgent(TaxiEnvPPONDNFBasedAgent):
    """
    An agent for gymnasium taxi environment, with `NeuralDNFMutexTanh` actor.
    This agent is used for training. It can be converted to a
    `TaxiEnvPPONDNFAgent` for post-training evaluation, or used directly for
    evaluation.
    To create a `TaxiEnvPPONDNFMTAgent` agent, pass in the following parameters:
    - num_latent (int): the number of conjunctions allowed in the NDNF-MT
    - use_decode_obs (bool): flag to use decode observation
    """

    actor: NeuralDNFMutexTanh

    def _create_default_actor(self) -> NeuralDNFMutexTanh:
        return NeuralDNFMutexTanh(
            self.num_inputs, self.num_latent, self.action_size, 1.0
        )

    def get_action_and_value(
        self, preprocessed_obs: dict[str, Tensor], action=None
    ) -> tuple[Tensor, Any, Any, Tensor]:
        """
        Return the action, log probability of the action, entropy of the action
        distribution, and the value of the state.
        This function is used in PPO algorithm
        """
        x = preprocessed_obs[self.input_key]
        logits = self.actor(x)
        dist = torch.distributions.Categorical(probs=(logits + 1) / 2)

        if action is None:
            action = dist.sample()

        return (
            action,
            dist.log_prob(action),
            dist.entropy(),
            self.critic(x),
        )

    def get_actor_output(
        self,
        preprocessed_obs: dict[str, Tensor],
        raw_output: bool = True,
        mutex_tanh: bool = False,
    ) -> Tensor:
        """
        Return the raw output of the `NeuralDNFMutexTanh` actor:
        - `raw_output` True: return the raw logits
        - `mutex_tanh` True: return the mutex-tanhed output
        This function should only be called during evaluation.
        """
        assert raw_output or mutex_tanh, "At least one of raw_output and "
        "mutex_tanh should be True!"

        assert not (raw_output and mutex_tanh), "Only one of raw_output and "
        "mutex_tanh can be True!"

        with torch.no_grad():
            x = preprocessed_obs[self.input_key]

        if raw_output:
            return self.actor.get_raw_output(x)
        return self.actor(x)

    def get_actions(
        self,
        preprocessed_obs: dict[str, Tensor],
        use_argmax: bool = True,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        """
        This function should only be called during evaluation.
        Because of the use of neural DNF module, the output of the actor can be
        treated as a symbolic output after tanh. This function returns both the
        probabilistic/argmax based action and the tanh action.
        """
        assert (
            not self.training
        ), "get_actions() should only be called during evaluation!"

        with torch.no_grad():
            x = preprocessed_obs[self.input_key]
            act = self.actor(x)
            dist = torch.distributions.Categorical(probs=(act + 1) / 2)
            tanh_actions = torch.tanh(self.actor.get_raw_output(x))

        actions = dist.probs.max(dim=1)[1] if use_argmax else dist.sample()  # type: ignore
        tanh_actions = torch.tanh(self.actor.get_raw_output(x))
        return (
            actions.detach().cpu().numpy(),
            tanh_actions.detach().cpu().numpy(),
        )

    def get_action_distribution(
        self, preprocessed_obs: dict[str, Tensor]
    ) -> Categorical:
        """
        Return the action distribution based on the observation.
        """
        x = preprocessed_obs[self.input_key]
        logits = self.actor(x)
        return Categorical(probs=(logits + 1) / 2)

    def get_aux_loss(
        self, preprocessed_obs: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """
        Return the auxiliary loss dictionary for the agent.
        The keys are:
        - l_disj_l1_mod: disjunction weight regularisation loss
        - l_tanh_conj: tanh conjunction output regularisation loss
        - l_mt_ce2: mutux tanh auxiliary loss
        """
        aux_loss_dict = super().get_aux_loss(preprocessed_obs)

        x = preprocessed_obs[self.input_key]

        act_out = self.actor(x)
        tanh_out = torch.tanh(self.actor.get_raw_output(x))

        p_k = (act_out + 1) / 2
        p_k_hat = (tanh_out + 1) / 2
        l_mt_ce2 = -torch.sum(
            p_k * torch.log(p_k_hat + 1e-8)
            + (1 - p_k) * torch.log(1 - p_k_hat + 1e-8)
        )

        return {
            **aux_loss_dict,
            "l_mt_ce2": l_mt_ce2,
        }


def construct_model(
    num_latent: int,
    use_ndnf: bool,
    use_decode_obs: bool,
    use_eo: bool = False,
    use_mt: bool = False,
    pretrained_critic: dict | None = None,
) -> TaxiEnvPPOBaseAgent:
    if not use_ndnf:
        return TaxiEnvPPOMLPAgent(num_latent, use_decode_obs)

    assert not (
        use_eo and use_mt
    ), "EO constraint and Mutex Tanh mode should not be active together."

    if not use_eo and not use_mt:
        agent = TaxiEnvPPONDNFAgent(num_latent, use_decode_obs)
    elif use_eo and not use_mt:
        agent = TaxiEnvPPONDNFEOAgent(num_latent, use_decode_obs)
    else:
        agent = TaxiEnvPPONDNFMTAgent(num_latent, use_decode_obs)

    if pretrained_critic is not None:
        agent.load_critic_from_trained_model(
            pretrained_critic["model_path"],
            pretrained_critic["disable_critic_training"],
        )

    return agent


def construct_single_environment(
    render_mode: str | None = "rgb_array",
) -> TaxiEnv:
    env = gym.make("Taxi-v3", render_mode=render_mode)
    return env  # type: ignore


def make_env(seed: int, idx: int, capture_video: bool):
    def thunk():
        if capture_video and idx == 0:
            env = construct_single_environment()
            video_dir = Path("videos")
            env = RecordVideo(env, str(video_dir.absolute()))
        else:
            env = construct_single_environment(None)
        env = RecordEpisodeStatistics(env)

        env.action_space.seed(seed)
        return env

    return thunk


def taxi_env_preprocess_obs(
    obs: np.ndarray,
    use_ndnf: bool,
    device: torch.device,
) -> dict[str, Tensor]:
    # obs: (num_envs, )
    input = F.one_hot(
        torch.tensor(obs),
        num_classes=N_OBSERVATION_SIZE,
    ).float()
    input = input.to(device)

    decode_input = torch.from_numpy(np.stack([decode(o) for o in obs])).float()
    decode_input = decode_input.to(device)

    if use_ndnf:
        input = torch.where(input == 0, -1.0, input)
        decode_input = torch.where(decode_input == 0, -1.0, decode_input)

    return {"input": input, "decode_input": decode_input}


def train_ppo(
    training_cfg: DictConfig,
    full_experiment_name: str,
    use_wandb: bool,
    writer: SummaryWriter,
    save_model: bool = True,
) -> tuple[Path | None, TaxiEnvPPOBaseAgent]:
    use_ndnf = "ndnf" in full_experiment_name
    use_decode_obs = training_cfg["use_decode_obs"]
    batch_size = int(training_cfg["num_envs"] * training_cfg["num_steps"])
    minibatch_size = int(batch_size // training_cfg["num_minibatches"])
    num_iterations = int(training_cfg["total_timesteps"] // batch_size)

    use_cuda = torch.cuda.is_available() and training_cfg["use_cuda"]

    use_mps = (
        torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
        and training_cfg.get("use_mps", False)
    )

    if use_mps:
        device = torch.device("mps")
    else:
        device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )

    print(f"Training on device: {device}")

    # Env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(i, i, False) for i in range(training_cfg["num_envs"])],
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    agent = construct_model(
        num_latent=training_cfg["num_latent_size"],
        use_ndnf=use_ndnf,
        use_decode_obs=use_decode_obs,
        use_eo="use_eo" in training_cfg and training_cfg["use_eo"],
        use_mt="use_mt" in training_cfg and training_cfg["use_mt"],
    )
    agent.train()
    agent.to(device)

    optimizer = optim.Adam(
        agent.parameters(), lr=training_cfg["learning_rate"], eps=1e-5
    )

    # ALGO Logic: Storage setup
    num_steps: int = training_cfg["num_steps"]
    num_envs: int = training_cfg["num_envs"]

    num_inputs = agent.num_inputs

    obs_shape = (num_steps, num_envs, num_inputs)
    obs = torch.zeros(obs_shape).to(device)

    action_shape = (num_steps, num_envs) + envs.single_action_space.shape  # type: ignore
    # action_shape: (num_steps, num_envs)
    actions = torch.zeros(action_shape).to(device)

    log_probs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_obs_dict = taxi_env_preprocess_obs(next_obs, use_ndnf, device)
    next_done = torch.zeros(num_envs).to(device)

    if isinstance(agent, TaxiEnvPPONDNFMTAgent):
        dds_cfg = training_cfg["dds"]
        dds = DeltaDelayedExponentialDecayScheduler(
            initial_delta=dds_cfg["initial_delta"],
            delta_decay_delay=dds_cfg["delta_decay_delay"],
            delta_decay_steps=dds_cfg["delta_decay_steps"],
            delta_decay_rate=dds_cfg["delta_decay_rate"],
            target_module_type=agent.actor.__class__.__name__,
        )
        agent.actor.set_delta_val(dds_cfg["initial_delta"])

    for iteration in range(1, num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if training_cfg["anneal_lr"]:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            lr_now = frac * training_cfg["learning_rate"]
            optimizer.param_groups[0]["lr"] = lr_now

        for step in range(0, num_steps):
            global_step += num_envs
            obs[step] = next_obs_dict[agent.input_key]
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(
                    next_obs_dict
                )
                values[step] = value.flatten()
            actions[step] = action
            log_probs[step] = log_prob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs_dict = taxi_env_preprocess_obs(next_obs, use_ndnf, device)
            next_done = torch.Tensor(next_done).to(device)

            last_episodic_return = None

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        if (
                            iteration % training_cfg["log_interval"] == 0
                            and step == num_steps - 1
                        ):
                            print(
                                f"global_step={global_step}, episodic_return={info['episode']['r']}"
                            )
                        writer.add_scalar(
                            "charts/episodic_return",
                            info["episode"]["r"],
                            global_step,
                        )
                        last_episodic_return = info["episode"]["r"]
                        writer.add_scalar(
                            "charts/episodic_length",
                            info["episode"]["l"],
                            global_step,
                        )

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs_dict).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            last_gae_lam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    next_non_terminal = 1.0 - next_done
                    next_vals = next_value
                else:
                    next_non_terminal = 1.0 - dones[t + 1]
                    next_vals = values[t + 1]
                delta = (
                    rewards[t]
                    + training_cfg["gamma"] * next_vals * next_non_terminal
                    - values[t]
                )
                advantages[t] = last_gae_lam = (
                    delta
                    + training_cfg["gamma"]
                    * training_cfg["gae_lambda"]
                    * next_non_terminal
                    * last_gae_lam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1, num_inputs))  # type: ignore
        b_log_probs = log_probs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)  # type: ignore
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clip_fracs = []
        for _ in range(training_cfg["update_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, new_log_prob, entropy, new_val = agent.get_action_and_value(
                    {agent.input_key: b_obs[mb_inds]}, b_actions.long()[mb_inds]
                )
                log_ratio = new_log_prob - b_log_probs[mb_inds]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fracs += [
                        ((ratio - 1.0).abs() > training_cfg["clip_coef"])
                        .float()
                        .mean()
                        .item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if training_cfg["norm_adv"]:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio,
                    1 - training_cfg["clip_coef"],
                    1 + training_cfg["clip_coef"],
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                new_val = new_val.view(-1)
                if training_cfg["clip_vloss"]:
                    v_loss_unclipped = (new_val - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        new_val - b_values[mb_inds],
                        -training_cfg["clip_coef"],
                        training_cfg["clip_coef"],
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((new_val - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    - training_cfg["ent_coef"] * entropy_loss
                    + v_loss * training_cfg["vf_coef"]
                )

                optimizer.zero_grad(set_to_none=True)

                if isinstance(agent, TaxiEnvPPONDNFBasedAgent):
                    aux_loss_dict = agent.get_aux_loss(next_obs_dict)

                    l_disj_l1_mod_lambda = training_cfg["aux_loss"][
                        "dis_l1_mod_lambda"
                    ]
                    l_disj_l1_mod = aux_loss_dict["l_disj_l1_mod"]
                    loss += l_disj_l1_mod_lambda * l_disj_l1_mod

                    l_tanh_conj_lambda = training_cfg["aux_loss"][
                        "tanh_conj_lambda"
                    ]
                    l_tanh_conj = aux_loss_dict["l_tanh_conj"]
                    loss += l_tanh_conj_lambda * l_tanh_conj

                    if isinstance(agent, TaxiEnvPPONDNFMTAgent):
                        l_mt_ce2_lambda = training_cfg["aux_loss"][
                            "mt_ce2_lambda"
                        ]
                        l_mt_ce2 = aux_loss_dict["l_mt_ce2"]
                        loss += l_mt_ce2_lambda * l_mt_ce2

                loss.backward()
                nn.utils.clip_grad_norm_(  # type: ignore
                    agent.parameters(), training_cfg["max_grad_norm"]
                )
                optimizer.step()

            if (
                training_cfg["target_kl"] is not None
                and approx_kl > training_cfg["target_kl"]  # type: ignore
            ):
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = (
            np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y  # type: ignore
        )

        if isinstance(agent, TaxiEnvPPONDNFBasedAgent):
            delta_dict = dds.step(agent.actor)
            new_delta = delta_dict["new_delta_vals"][0]
            old_delta = delta_dict["old_delta_vals"][0]

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)  # type: ignore
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)  # type: ignore
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)  # type: ignore
        writer.add_scalar(
            "losses/old_approx_kl", old_approx_kl.item(), global_step  # type: ignore
        )
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)  # type: ignore
        writer.add_scalar("losses/clipfrac", np.mean(clip_fracs), global_step)
        writer.add_scalar(
            "losses/explained_variance", explained_var, global_step
        )
        if isinstance(agent, TaxiEnvPPONDNFMTAgent):
            writer.add_scalar(
                "losses/l_disj_l1_mod", l_disj_l1_mod.item(), global_step
            )
            writer.add_scalar(
                "losses/l_tanh_conj", l_tanh_conj.item(), global_step
            )

            if isinstance(agent, TaxiEnvPPONDNFMTAgent):
                writer.add_scalar(
                    "losses/l_mt_ce2", l_mt_ce2.item(), global_step
                )

        if isinstance(agent, TaxiEnvPPONDNFMTAgent):
            writer.add_scalar("charts/delta", old_delta, global_step)  # type: ignore
            if new_delta != old_delta:  # type: ignore
                print(
                    f"i={iteration}\t"
                    f"old delta={old_delta:.3f} new delta={new_delta:.3f}\t"
                    f"last episodic_return={last_episodic_return}"
                )  # type: ignore

        writer.add_scalar(
            "charts/SPS",
            int(global_step / (time.time() - start_time)),
            global_step,
        )

    envs.close()
    writer.close()

    if save_model:
        model_dir = PPO_MODEL_DIR / full_experiment_name
        if not model_dir.exists() or not model_dir.is_dir():
            model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pth"
        torch.save(agent.state_dict(), model_path)
        return model_path, agent

    return None, agent


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiment(cfg: DictConfig):
    training_cfg = cfg["training"]
    seed = training_cfg["seed"]

    full_experiment_name = f"{training_cfg['experiment_name']}_{seed}"

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Expect the experiment name to be in the format of
    # taxi_ppo_{model type}_...
    name_list = training_cfg["experiment_name"].split("_")
    # Insert "raw"/"dec" after model type to indicate whether to use
    # the raw observation or the decoded observation
    name_list.insert(3, "dec" if training_cfg["use_decode_obs"] else "raw")
    name_list.append(str(seed))
    full_experiment_name = "_".join(name_list)

    # For wandb and output dir name, capitalise the first 2 words:
    # 'taxi' and 'ppo'
    run_dir_name = "-".join(
        [(s.upper() if i in [0, 1] else s) for i, s in enumerate(name_list)]
    )

    use_wandb = cfg["wandb"]["use_wandb"]
    if use_wandb:
        run = wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"]["entity"],
            config=OmegaConf.to_container(
                training_cfg, resolve=True, throw_on_missing=True
            ),  # type: ignore
            dir=HydraConfig.get().run.dir,
            name=run_dir_name,
            tags=cfg["wandb"]["tags"] if "tags" in cfg["wandb"] else [],
            monitor_gym=True,
            save_code=True,
            sync_tensorboard=True,
            group=cfg["wandb"]["group"] if "group" in cfg["wandb"] else None,
        )

    # torch.autograd.set_detect_anomaly(True)  # type: ignore

    writer_dir = Path(HydraConfig.get().run.dir) / "tb"
    writer = SummaryWriter(writer_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % (
            "\n".join(
                [
                    f"|{key}|{value}|"
                    for key, value in vars(training_cfg).items()
                ]
            )
        ),
    )

    use_discord_webhook = cfg["webhook"]["use_discord_webhook"]
    msg_body = None
    errored = False
    keyboard_interrupt = None

    try:
        model_path, _ = train_ppo(
            training_cfg, full_experiment_name, use_wandb, writer
        )
        assert model_path is not None
        if use_wandb:
            wandb.save(glob_str=str(model_path.absolute()))

        if use_discord_webhook:
            msg_body = "Success!"
    except BaseException as e:
        if use_discord_webhook:
            if isinstance(e, KeyboardInterrupt):
                keyboard_interrupt = True
            else:
                msg_body = "Check the logs for more details."

        print(traceback.format_exc())
        errored = True
    finally:
        if use_discord_webhook:
            if msg_body is None:
                msg_body = ""
            webhook_url = cfg["webhook"]["discord_webhook_url"]
            post_to_discord_webhook(
                webhook_url=webhook_url,
                experiment_name=full_experiment_name,
                message_body=msg_body,
                errored=errored,
                keyboard_interrupt=keyboard_interrupt,
            )
        if use_wandb:
            wandb.finish()
        if not errored:
            path = Path(HydraConfig.get().run.dir)
            path.rename(path.absolute().parent / run_dir_name)


if __name__ == "__main__":
    import multiprocessing as mp

    if mp.get_start_method() != "forkserver":
        mp.set_start_method("forkserver", force=True)

    run_experiment()
