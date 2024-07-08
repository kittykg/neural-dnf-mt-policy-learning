from pathlib import Path
import random
import time
import traceback
from typing import Any


import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig, OmegaConf
import torch
from torch import Tensor, nn, optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter  # type: ignore
import wandb

from corridor_grid.envs.base_ss_corridor import BaseSpecialStateCorridorEnv
from corridor_grid.envs import (
    SmallSSCorridorEnv,
    LongSSCorridorEnv,
    CircularSSCorridorEnv,
)

from neural_dnf import NeuralDNF, NeuralDNFEO, NeuralDNFMutexTanh
from neural_dnf.neural_dnf import BaseNeuralDNF  # for type hinting
from neural_dnf.utils import DeltaDelayedExponentialDecayScheduler

from common import init_params
from utils import post_to_discord_webhook


PPO_MODEL_DIR = Path(__file__).parent / "ssc_ppo_storage/"
if not PPO_MODEL_DIR.exists() or not PPO_MODEL_DIR.is_dir():
    PPO_MODEL_DIR.mkdir(parents=True, exist_ok=True)


class SSCPPOBaseAgent(nn.Module):
    """
    To create a base agent, pass in the following parameters:
    - num_inputs (int): the number of input features
    - num_latent (int): the number of latent features
    - action_size (int): the number of actions the agent can take

    The actor and critic networks are created using `_create_default_actor()`
    and `_create_default_critic()` methods respectively.
    """

    # Model components
    actor: nn.Module
    critic: nn.Module

    # Actor parameters
    num_inputs: int
    num_latent: int
    action_size: int

    def __init__(
        self,
        num_inputs: int,
        num_latent: int,
        action_size: int,
    ) -> None:
        super().__init__()

        self.num_inputs = num_inputs
        self.num_latent = num_latent
        self.action_size = action_size

        self.actor = self._create_default_actor()
        self.critic = self._create_default_critic()

        self._init_params()

    def get_value(self, preprocessed_obs: dict[str, Tensor]) -> Tensor:
        """
        Return the value of the state.
        This function is used in PPO algorithm
        """
        return self.critic(preprocessed_obs["input"])

    def get_action_and_value(
        self, preprocessed_obs: dict[str, Tensor], action=None
    ) -> tuple[Tensor, Any, Any, Tensor]:
        """
        Return the action, log probability of the action, entropy of the action
        distribution, and the value of the state.
        This function is used in PPO algorithm
        """
        x = preprocessed_obs["input"]
        logits = self.actor(x)
        dist = Categorical(logits=logits)

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
        x = preprocessed_obs["input"]
        logits = self.actor(x)
        dist = Categorical(logits=logits)

        actions = dist.probs.max(dim=1)[1] if use_argmax else dist.sample()  # type: ignore
        return actions.cpu().numpy()

    def get_action_distribution(
        self, preprocessed_obs: dict[str, Tensor]
    ) -> Categorical:
        """
        Return the action distribution based on the observation.
        """
        x = preprocessed_obs["input"]
        logits = self.actor(x)
        return Categorical(logits=logits)

    def _create_default_actor(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.num_inputs, self.num_latent),
            nn.Tanh(),
            nn.Linear(self.num_latent, self.action_size),
        )

    def _create_default_critic(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.num_inputs, 64), nn.Tanh(), nn.Linear(64, 1)
        )

    def _init_params(self) -> None:
        self.apply(init_params)


class SSCPPOMLPAgent(SSCPPOBaseAgent):
    """
    An agent for special state corridor environment, with a 2-layer MLP actor.
    To create a `SSCPPOMLP` agent, pass in the following parameters:
    - num_inputs (int): the number of input features
    - num_latent (int): the number of latent features
    - action_size (int): the number of actions the agent can take

    The actor and critic networks are created using `_create_default_actor()`
    and `_create_default_critic()` methods respectively.
    """


class SSCPPONDNFBasedAgent(SSCPPOBaseAgent):
    """
    Base class for agents using a neural DNF module as the actor.
    """

    actor: BaseNeuralDNF

    def _create_default_actor(self) -> nn.Module:
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
        x = preprocessed_obs["input"]
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
            x = preprocessed_obs["input"]
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
        dist = Categorical(logits=raw_actions)
        if use_argmax:
            actions = dist.probs.max(1)[1]  # type: ignore
        else:
            actions = dist.sample()
        tanh_action = torch.tanh(raw_actions)

        return actions.cpu().numpy(), tanh_action.cpu().numpy()


class SSCPPONDNFAgent(SSCPPONDNFBasedAgent):
    """
    An agent for special state corridor environment, with `NeuralDNF` as actor.
    This agent is not usually expected to use for training, but if treating the
    2 actions as binary classification (go left; not go left which is go right),
    it can be used for training. This agent is more expected to be used as a
    post-training evaluation agent from either a trained `SSCPPONDNFEOAgent` or
    `SSCPPONDNFMutexTanhAgent`.
    To create a `SSCPPONDNFEOAgent` agent, pass in the following parameters:
    - num_inputs (int): the number of input features
    - num_latent (int): the number of conjunctions allowed in NDNF-EO
    - action_size (int): the number of actions the agent can take
    """

    actor: NeuralDNF

    def _create_default_actor(self) -> nn.Module:
        return NeuralDNF(
            self.num_inputs, self.num_latent, self.action_size, 1.0
        )


class SSCPPONDNFEOAgent(SSCPPONDNFBasedAgent):
    """
    An agent for special state corridor environment, with `NeuralDNFEO` actor.
    This agent is used for training, and to be converted to a `SSCPPONDNFAgent`
    for post-training evaluation.
    To create a `SSCPPONDNFEOAgent` agent, pass in the following parameters:
    - num_inputs (int): the number of input features
    - num_latent (int): the number of conjunctions allowed in NDNF-EO
    - action_size (int): the number of actions the agent can take
    """

    actor: NeuralDNFEO

    def _create_default_actor(self) -> nn.Module:
        return NeuralDNFEO(
            self.num_inputs, self.num_latent, self.action_size, 1.0
        )

    def to_ndnf_agent(self) -> SSCPPONDNFAgent:
        """
        Convert this agent to a SSCPPONDNFAgent.
        """
        ndnf_agent = SSCPPONDNFAgent(
            self.num_inputs, self.num_latent, self.action_size
        )
        ndnf_agent.actor = self.actor.to_ndnf()
        return ndnf_agent


class SSCPPONDNFMutexTanhAgent(SSCPPONDNFBasedAgent):
    """
    An agent for special state corridor environment, with `NeuralDNFMutexTanh`
    actor.
    This agent is used for training. It can be converted to a `SSCPPONDNFAgent`
    for post-training evaluation, or used directly for evaluation.
    To create a `SSCPPONDNFMutexTanhAgent` agent, pass in the following parameters:
    - num_inputs (int): the number of input features
    - num_latent (int): the number of conjunctions allowed in the NDNF-MT
    - action_size (int): the number of actions the agent can take
    """

    actor: NeuralDNFMutexTanh

    def _create_default_actor(self) -> nn.Module:
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
        x = preprocessed_obs["input"]
        logits = self.actor(x)
        dist = Categorical(probs=(logits + 1) / 2)

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
            x = preprocessed_obs["input"]

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
            x = preprocessed_obs["input"]
            act = self.actor(x)
            dist = Categorical(probs=(act + 1) / 2)
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
        x = preprocessed_obs["input"]
        act = self.actor(x)
        return Categorical(probs=(act + 1) / 2)

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

        x = preprocessed_obs["input"]

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


def ss_corridor_preprocess_obs(
    use_state_no_as_obs: bool,
    use_ndnf: bool,
    corridor_length: int,
    obs: dict[str, np.ndarray],
    device: torch.device | None = None,
) -> Tensor:
    if use_state_no_as_obs:
        obs_s = obs["agent_location"]
        state = np.zeros((len(obs_s), corridor_length), dtype=np.float32)
        state[range(len(obs_s)), obs["agent_location"]] = 1
    else:
        state = np.array(obs["wall_status"], dtype=np.float32)

    if use_ndnf:
        state = np.where(state == 0, -1, state)

    return torch.Tensor(state, device=device)


def construct_model(
    num_inputs: int,
    num_latent: int,
    action_size: int,
    use_ndnf: bool,
    use_eo: bool = False,
    use_mt: bool = False,
) -> SSCPPOBaseAgent:
    if not use_ndnf:
        return SSCPPOMLPAgent(num_inputs, num_latent, action_size)

    assert not (
        use_eo and action_size == 1
    ), "EO constraint should not be active if there's only one action."
    assert not (
        use_mt and action_size == 1
    ), "Mutex Tanh mode should not be active if there's only one action."
    assert not (
        use_eo and use_mt
    ), "EO constraint and Mutex Tanh mode should not be active together."

    if not use_eo and not use_mt:
        return SSCPPONDNFAgent(num_inputs, num_latent, action_size)
    if use_eo and not use_mt:
        return SSCPPONDNFEOAgent(num_inputs, num_latent, action_size)
    return SSCPPONDNFMutexTanhAgent(num_inputs, num_latent, action_size)


def construct_single_environment(
    cfg: DictConfig, render_mode: str = "ansi"
) -> BaseSpecialStateCorridorEnv:
    if "long_corridor" in cfg:
        env = LongSSCorridorEnv(
            render_mode=render_mode,
            customisation_cfg_dict=cfg["long_corridor"],
        )
    elif "circular_corridor" in cfg:
        env = CircularSSCorridorEnv(
            render_mode=render_mode,
            customisation_cfg_dict=cfg["circular_corridor"],
        )
    else:
        env = SmallSSCorridorEnv(render_mode=render_mode)
    return env


def make_env(cfg: DictConfig, seed: int, idx: int, capture_video: bool):
    def thunk():
        if capture_video and idx == 0:
            env = construct_single_environment(cfg, render_mode="rgb_array")
            video_dir = Path("videos")
            env = RecordVideo(env, str(video_dir.absolute()))
        else:
            env = construct_single_environment(cfg)
        env = RecordEpisodeStatistics(env)

        env.action_space.seed(seed)
        return env

    return thunk


def train_ppo(
    training_cfg: DictConfig,
    full_experiment_name: str,
    writer: SummaryWriter,
    save_model: bool = True,
) -> tuple[Path | None, SSCPPOBaseAgent]:
    use_ndnf = "ndnf" in full_experiment_name
    use_state_no_as_obs = training_cfg["use_state_no_as_obs"]
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
        [
            make_env(training_cfg, i, i, False)
            for i in range(training_cfg["num_envs"])
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    # Set up the model
    single_env = construct_single_environment(training_cfg)
    num_inputs = single_env.corridor_length if use_state_no_as_obs else 2
    agent = construct_model(
        num_inputs=num_inputs,
        num_latent=training_cfg["model_latent_size"],
        action_size=int(envs.single_action_space.n),
        use_ndnf=use_ndnf,
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
    process_obs = lambda obs: ss_corridor_preprocess_obs(
        use_state_no_as_obs=use_state_no_as_obs,
        use_ndnf=use_ndnf,
        corridor_length=single_env.corridor_length,
        obs=obs,
        device=device,
    )
    global_step = 0
    start_time = time.time()
    next_obs_dict, _ = envs.reset()
    next_obs = process_obs(next_obs_dict)
    next_obs_dict = {"input": next_obs}
    next_done = torch.zeros(num_envs).to(device)

    last_episodic_return = None

    if isinstance(agent, SSCPPONDNFBasedAgent):
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
            obs[step] = next_obs
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
            next_obs_dict, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).float().to(device).view(-1)
            next_obs = process_obs(next_obs_dict)
            next_obs_dict = {"input": next_obs}
            next_done = torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        if (
                            iteration % training_cfg["log_interval"] == 0
                            and step == num_steps - 1
                        ):
                            print(
                                f"global_step={global_step}, "
                                f"episodic_return={info['episode']['r']}"
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
        b_obs = obs.reshape((-1, num_inputs))
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
                    {"input": b_obs[mb_inds]}, b_actions.long()[mb_inds]
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

                if isinstance(agent, SSCPPONDNFBasedAgent):
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

                    if isinstance(agent, SSCPPONDNFMutexTanhAgent):
                        l_tanh_conj_lambda = training_cfg["aux_loss"][
                            "tanh_conj_lambda"
                        ]
                        l_tanh_conj = aux_loss_dict["l_tanh_conj"]
                        loss += l_tanh_conj_lambda * l_tanh_conj

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

        if isinstance(agent, SSCPPONDNFBasedAgent):
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
        if isinstance(agent, SSCPPONDNFBasedAgent):
            writer.add_scalar(
                "losses/l_disj_l1_mod", l_disj_l1_mod.item(), global_step
            )
            writer.add_scalar(
                "losses/l_tanh_conj", l_tanh_conj.item(), global_step
            )

            if isinstance(agent, SSCPPONDNFMutexTanhAgent):
                writer.add_scalar(
                    "losses/l_mt_ce2", l_mt_ce2.item(), global_step
                )

        if isinstance(agent, SSCPPONDNFBasedAgent):
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

        # If training SSCPPONeuralDNFMutexTanhAgent, plot y_k, hat_y_k
        if isinstance(agent, SSCPPONDNFMutexTanhAgent):
            ret = plot_y_k_hat_y_k(agent, single_env, use_state_no_as_obs)
            for key, value in ret.items():
                writer.add_scalar(f"mt/{key}", value, global_step)

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


def plot_y_k_hat_y_k(
    model: SSCPPONDNFMutexTanhAgent,
    single_env: BaseSpecialStateCorridorEnv,
    use_state_no_as_obs: bool,
) -> dict[str, Any]:
    if use_state_no_as_obs:
        # Take state 0 and one of the special states
        state = np.zeros((2, single_env.corridor_length), dtype=np.float32)
        state[[0, 1], [0, single_env.special_states[0]]] = 1
    else:
        # Take all possible wall status
        # Note that [1, 0] or [0, 1] might never be seen in the circular
        # corridor. If not circular, the agent might never see [1, 0] or [0, 1]
        # if the start/goal is not at the edge.
        state = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    state = np.where(state == 0, -1, state)

    obs = {
        "input": torch.Tensor(
            state, device=next(model.actor.parameters()).device
        )
    }

    with torch.no_grad():
        hat_y = torch.tanh(
            model.get_actor_output(
                preprocessed_obs=obs, raw_output=True, mutex_tanh=False
            )
        )
        y = model.get_actor_output(
            preprocessed_obs=obs, raw_output=False, mutex_tanh=True
        )

    ret = {}
    for i in range(len(obs)):
        for j in range(hat_y.shape[1]):
            ret[f"y_{i}_{j}"] = y[i, j].item()
            ret[f"hat_y_{i}_{j}"] = hat_y[i, j].item()

    return ret


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    training_cfg = cfg["training"]
    seed = training_cfg["seed"]

    # Expect the experiment name to be in the format of
    # {corridor env name}_ppo_{model type}_...
    name_list = training_cfg["experiment_name"].split("_")
    # Insert "sn"/"ws" between corridor env name and ppo to indicate whether the
    # environment is POMDP or MDP
    name_list.insert(1, "sn" if training_cfg["use_state_no_as_obs"] else "ws")
    # Add seed at the end of the name list
    name_list.append(str(seed))
    full_experiment_name = "_".join(name_list)

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # For wandb and output dir name, capitalise the first 3 words:
    # corridor env name, sn/ws, ppo
    run_dir_name = "-".join(
        [(s.upper() if i in [0, 1, 2] else s) for i, s in enumerate(name_list)]
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
        model_path, _ = train_ppo(training_cfg, full_experiment_name, writer)
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
