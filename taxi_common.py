from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium.envs.toy_text.taxi import TaxiEnv
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F

from common import init_params
from neural_dnf import NeuralDNF, NeuralDNFEO, NeuralDNFMutexTanh
from neural_dnf.neural_dnf import BaseNeuralDNF  # for type hinting

N_OBSERVATION_SIZE: int = 500
TUPLE_LIMITS_LIST: list[int] = [5, 5, 5, 4]
N_DECODE_OBSERVATION_SIZE: int = sum(TUPLE_LIMITS_LIST)
N_ACTIONS: int = 6


# =============================================================================#
#                                    Agent                                     #
# =============================================================================#


class TaxiEnvPPOBaseAgent(nn.Module):
    """
    To create a base agent, pass in the following parameters:
    - num_latent: The latent size of the agent
    - use_decode_obs: Whether to use the decoded observation or not
    """

    # Model components
    actor: nn.Module  # 2 layer MLP or NeuralDNF
    critic: nn.Sequential

    # Actor parameters
    # 2 layers: num_inputs x actor_latent -> actor_latent x N_ACTIONS
    num_inputs: int  # calculated based on `use_decode_obs`
    actor_latent_size: int
    action_size: int = N_ACTIONS
    share_layer_with_critic: bool

    # Critic parameters
    # 3 layers (if not sharing with actor):
    #   num_inputs x critic_latent_1 ->
    #   critic_latent_1 x critic_latent_2 ->
    #   critic_latent_2 x 1
    # 2 layers (if sharing with actor):
    #   num_inputs x actor_latent (from actor) ->
    #   actor_latent x critic_latent_1 ->
    #   critic_latent_1 x 1
    critic_latent_1: int
    critic_latent_2: int | None

    # Flag to use decode observation
    use_decode_obs: bool
    input_key: str

    def __init__(
        self,
        use_decode_obs: bool,
        actor_latent_size: int,
        share_layer_with_critic: bool,
        critic_latent_1: int,
        critic_latent_2: int | None,
    ):
        super().__init__()
        self.actor_latent_size = actor_latent_size
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

        self.share_layer_with_critic = share_layer_with_critic

        self.critic_latent_1 = critic_latent_1
        self.critic_latent_2 = critic_latent_2

        self.actor = self._create_default_actor()
        self.critic = self._create_default_critic()

        self._init_params()

    def get_value(self, preprocessed_obs: dict[str, Tensor]) -> Tensor:
        """
        Return the value of the state.
        This function is used in PPO algorithm
        """
        if not self.share_layer_with_critic:
            return self.critic(preprocessed_obs[self.input_key])

        x = preprocessed_obs[self.input_key]
        x = self._get_actor_first_layer_output(x)
        return self.critic(x)

    def _get_actor_first_layer_output(self, x: Tensor) -> Tensor:
        """
        Return the output of the actor's first layer, if the critic and actor
        shares the first layer.
        """
        raise NotImplementedError

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
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        return (
            action,
            dist.log_prob(action),
            dist.entropy(),
            self.get_value(preprocessed_obs),
        )

    def get_actions(
        self, preprocessed_obs: dict[str, Tensor], use_argmax: bool = True
    ) -> npt.NDArray[np.int64]:
        """
        Return the actions based on the observation.
        """
        x = preprocessed_obs[self.input_key]
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
        x = preprocessed_obs[self.input_key]
        logits = self.actor(x)
        return Categorical(logits=logits)

    def _create_default_actor(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.num_inputs, self.actor_latent_size),
            nn.Tanh(),
            nn.Linear(self.actor_latent_size, self.action_size),
        )

    def _create_default_critic(self) -> nn.Sequential:
        if not self.share_layer_with_critic:
            assert self.critic_latent_2 is not None, "critic_latent_2 is None"

            return nn.Sequential(
                nn.Linear(self.num_inputs, self.critic_latent_1),
                nn.ReLU(),
                nn.Linear(self.critic_latent_1, self.critic_latent_2),
                nn.ReLU(),
                nn.Linear(self.critic_latent_2, 1),
            )

        return nn.Sequential(
            nn.Linear(self.actor_latent_size, self.critic_latent_1),
            nn.ReLU(),
            nn.Linear(self.critic_latent_1, 1),
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

    actor: nn.Sequential
    actor_disable_bias: bool

    def __init__(
        self,
        use_decode_obs: bool,
        actor_latent_size: int,
        share_layer_with_critic: bool,
        critic_latent_1: int,
        critic_latent_2: int | None,
        actor_disable_bias: bool = False,
    ):
        self.actor_disable_bias = actor_disable_bias

        super().__init__(
            use_decode_obs,
            actor_latent_size,
            share_layer_with_critic,
            critic_latent_1,
            critic_latent_2,
        )

    def _create_default_actor(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(
                self.num_inputs,
                self.actor_latent_size,
                bias=not self.actor_disable_bias,
            ),
            nn.Tanh(),
            nn.Linear(
                self.actor_latent_size,
                self.action_size,
                bias=not self.actor_disable_bias,
            ),
        )

    def _get_actor_first_layer_output(self, x: Tensor) -> Tensor:
        """
        Return the value of the state.
        This function is used in PPO algorithm and A2C algorithm
        """
        return torch.tanh(self.actor[0](x))


class TaxiEnvPPONDNFBasedAgent(TaxiEnvPPOBaseAgent):
    """
    Base class for agents using a neural DNF module as the actor.
    """

    actor: BaseNeuralDNF

    def _get_actor_first_layer_output(self, x: Tensor) -> Tensor:
        """
        Return the output of the actor's first layer, if the critic and actor
        shares the first layer.
        """
        return torch.tanh(self.actor.conjunctions(x))

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
        dist = Categorical(logits=raw_actions)
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
            self.num_inputs, self.actor_latent_size, self.action_size, 1.0
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
            self.num_inputs, self.actor_latent_size, self.action_size, 1.0
        )

    def to_ndnf_agent(self) -> TaxiEnvPPONDNFAgent:
        """
        Convert this agent to a TaxiEnvPPONDNFAgent.
        """
        ndnf_agent = TaxiEnvPPONDNFAgent(
            use_decode_obs=self.use_decode_obs,
            actor_latent_size=self.actor_latent_size,
            share_layer_with_critic=self.share_layer_with_critic,
            critic_latent_1=self.critic_latent_1,
            critic_latent_2=self.critic_latent_2,
        )
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
            self.num_inputs, self.actor_latent_size, self.action_size, 1.0
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
        dist = Categorical(probs=(logits + 1) / 2)

        if action is None:
            action = dist.sample()

        return (
            action,
            dist.log_prob(action),
            dist.entropy(),
            self.get_value(preprocessed_obs),
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
    actor_latent_size: int,
    use_ndnf: bool,
    use_decode_obs: bool,
    use_eo: bool = False,
    use_mt: bool = False,
    share_layer_with_critic: bool = False,
    critic_latent_1: int = 256,
    critic_latent_2: int | None = 64,
    pretrained_critic: dict | None = None,
    mlp_actor_disable_bias: bool = False,
) -> TaxiEnvPPOBaseAgent:
    if not use_ndnf:
        return TaxiEnvPPOMLPAgent(
            use_decode_obs=use_decode_obs,
            actor_latent_size=actor_latent_size,
            share_layer_with_critic=share_layer_with_critic,
            critic_latent_1=critic_latent_1,
            critic_latent_2=critic_latent_2,
            actor_disable_bias=mlp_actor_disable_bias,
        )

    assert not (
        use_eo and use_mt
    ), "EO constraint and Mutex Tanh mode should not be active together."

    if not use_eo and not use_mt:
        agent = TaxiEnvPPONDNFAgent(
            use_decode_obs=use_decode_obs,
            actor_latent_size=actor_latent_size,
            share_layer_with_critic=share_layer_with_critic,
            critic_latent_1=critic_latent_1,
            critic_latent_2=critic_latent_2,
        )
    elif use_eo and not use_mt:
        agent = TaxiEnvPPONDNFEOAgent(
            use_decode_obs=use_decode_obs,
            actor_latent_size=actor_latent_size,
            share_layer_with_critic=share_layer_with_critic,
            critic_latent_1=critic_latent_1,
            critic_latent_2=critic_latent_2,
        )
    else:
        agent = TaxiEnvPPONDNFMTAgent(
            use_decode_obs=use_decode_obs,
            actor_latent_size=actor_latent_size,
            share_layer_with_critic=share_layer_with_critic,
            critic_latent_1=critic_latent_1,
            critic_latent_2=critic_latent_2,
        )

    if pretrained_critic is not None:
        agent.load_critic_from_trained_model(
            pretrained_critic["model_path"],
            pretrained_critic["disable_critic_training"],
        )

    return agent


# =============================================================================#
#                            Environment construction                          #
# =============================================================================#


def construct_single_environment(
    render_mode: str | None = None,
) -> TaxiEnv:
    # Can be set to "rgb_array"
    env = gym.make("Taxi-v3", render_mode=render_mode)
    return env  # type: ignore


def make_env(action_space_seed: int, idx: int, capture_video: bool):
    def thunk():
        if capture_video and idx == 0:
            env = construct_single_environment()
            video_dir = Path("videos")
            env = RecordVideo(env, str(video_dir.absolute()))
        else:
            env = construct_single_environment(None)
        env = RecordEpisodeStatistics(env)

        env.action_space.seed(action_space_seed)
        return env

    return thunk


# =============================================================================#
#                               Data structures                                #
# =============================================================================#


class PassengerLocation(Enum):
    RED = (0, (0, 0))
    GREEN = (1, (0, 4))
    YELLOW = (2, (4, 0))
    BLUE = (3, (4, 3))
    TAXI = 4

    @classmethod
    def get_location_from_colour_index(cls, idx: int) -> tuple[int, int]:
        assert idx != cls.TAXI.value, "Taxi is not a colour"

        colour_loc_map: dict[int, tuple[int, int]] = dict(
            [
                cls.RED.value,
                cls.GREEN.value,
                cls.YELLOW.value,
                cls.BLUE.value,
            ]
        )

        return colour_loc_map[idx]


# =============================================================================#
#                             Observation processing                           #
# =============================================================================#


def convert_n_to_one_hot(obs: gym.spaces.Discrete | int) -> np.ndarray:
    """
    Convert the observation into a one-hot encoding.
    """
    return np.eye(N_OBSERVATION_SIZE)[obs]  # type: ignore


def encode(
    taxi_row: int, taxi_col: int, passenger_location: int, destination: int
) -> int:
    """
    Encode the (taxi_row, taxi_col, passenger_location, destination) into a
    single integer.
    """
    for i, j in zip(
        [taxi_row, taxi_col, passenger_location, destination], TUPLE_LIMITS_LIST
    ):
        assert 0 <= i < j, f"{i} going outside of [0, {j})"

    return (
        (taxi_row * 5 + taxi_col) * 5 + passenger_location
    ) * 4 + destination


def decode(obs: gym.spaces.Discrete | int) -> np.ndarray:
    """
    Decode the single integer into a concatenated one-hot encoding of
    (taxi_row, taxi_col, passenger_location, destination).
    """
    i = int(obs)  # type: ignore
    out: list[int] = []
    out.append(i % 4)
    i = i // 4
    out.append(i % 5)
    i = i // 5
    out.append(i % 5)
    i = i // 5
    out.append(i)
    assert 0 <= i < 5

    return np.concatenate(
        [np.eye(s)[o] for o, s in zip(reversed(out), [5, 5, 5, 4])]
    )


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


# =============================================================================#
#                              Policy comparison                               #
# =============================================================================#


def split_all_states_to_reachable_and_non() -> tuple[list[int], list[int]]:
    """
    In the taxi environment, only 404 out of 500 states are reachable, and 4 of
    the reachable states are terminal states.
    This function split all states into reachable (excluding 4 terminal states,
    400 in total) and non-reachable states (100 in total).
    For more details, see the taxi environment documentation:
    https://gymnasium.farama.org/environments/toy_text/taxi/
    """
    reachable = []
    non_reachable = []

    for x in range(TUPLE_LIMITS_LIST[0]):
        for y in range(TUPLE_LIMITS_LIST[1]):
            for p in range(TUPLE_LIMITS_LIST[2]):
                for d in range(TUPLE_LIMITS_LIST[3]):
                    if p == d:
                        non_reachable.append(encode(x, y, p, d))
                    else:
                        reachable.append(encode(x, y, p, d))

    assert len(reachable) == 400
    assert len(reachable) + len(non_reachable) == 500

    return reachable, non_reachable
