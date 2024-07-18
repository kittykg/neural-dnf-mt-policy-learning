from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any


import gymnasium as gym
from gymnasium.envs.toy_text.blackjack import BlackjackEnv
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from matplotlib.figure import Figure, figaspect
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
import pandas as pd
import seaborn as sns

from neural_dnf import NeuralDNF, NeuralDNFEO, NeuralDNFMutexTanh
from neural_dnf.neural_dnf import BaseNeuralDNF  # for type hinting

from common import init_params

BLACKJACK_TUPLE_OBS_FIRST_TWO = [32, 11]
N_ACTIONS = 2
N_OBSERVATION_DECODE_SIZE = 44  # 32 + 11 + 1
N_OBSERVATION_SIZE = 3


TargetQValueTableType = OrderedDict[tuple[int, int, int], dict[int, float]]
TargetPolicyType = OrderedDict[tuple[int, int, int], int]


# =============================================================================#
#                                    Agent                                     #
# =============================================================================#


class BlackjackBaseAgent(nn.Module):
    """
    To create a base agent, pass in the following parameters:
    - num_latent (int): the number of latent features
    - use_decode_obs (bool): flag to use decode observation

    The actor and critic networks are created using `_create_default_actor()`
    and `_create_default_critic()` methods respectively.
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

    def __init__(
        self,
        num_latent: int,
        use_decode_obs: bool,
    ) -> None:
        super().__init__()

        self.use_decode_obs = use_decode_obs
        if self.use_decode_obs:
            self.input_key = "decode_input"
        else:
            self.input_key = "input"
        self.num_inputs = (
            N_OBSERVATION_DECODE_SIZE if use_decode_obs else N_OBSERVATION_SIZE
        )
        self.num_latent = num_latent

        self.actor = self._create_default_actor()
        self.critic = self._create_default_critic()

        self._init_params()

    def get_value(self, preprocessed_obs: dict[str, Tensor]) -> Tensor:
        """
        Return the value of the state.
        This function is used in PPO algorithm and A2C algorithm
        """
        return self.critic(preprocessed_obs[self.input_key])

    def get_action_and_value(
        self, preprocessed_obs: dict[str, Tensor], action=None
    ) -> tuple[Tensor, Any, Any, Tensor]:
        """
        Return the action, log probability of the action, entropy of the action
        distribution, and the value of the state.
        This function is used in PPO algorithm and A2C algorithm
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
            self.critic(x),
        )

    def get_actions(
        self, preprocessed_obs: dict[str, Tensor], use_argmax: bool = True
    ) -> npt.NDArray:
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
        if self.use_decode_obs:
            return nn.Sequential(
                nn.Linear(N_OBSERVATION_DECODE_SIZE, self.num_latent),
                nn.Tanh(),
                nn.Linear(self.num_latent, self.action_size),
            )

        return nn.Sequential(
            nn.Linear(N_OBSERVATION_SIZE, 64),
            nn.Tanh(),
            nn.Linear(64, self.num_latent),
            nn.Tanh(),
            nn.Linear(self.num_latent, self.action_size),
        )

    def _create_default_critic(self) -> nn.Module:
        if self.use_decode_obs:
            return nn.Sequential(
                nn.Linear(self.num_inputs, 64), nn.Tanh(), nn.Linear(64, 1)
            )

        return nn.Sequential(
            nn.Linear(N_OBSERVATION_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def _init_params(self) -> None:
        self.apply(init_params)


class BlackjackMLPAgent(BlackjackBaseAgent):
    """
    An agent for gymnasium Blackjack environment, with a 2-layer MLP actor.
    To create a `BlackjackMLP` agent, pass in the following parameters:
    - num_latent (int): the number of latent features
    - use_decode_obs (bool): flag to use decode observation

    The actor and critic networks are created using `_create_default_actor()`
    and `_create_default_critic()` methods respectively.
    """


class BlackjackNDNFBasedAgent(BlackjackBaseAgent):
    """
    Base class for agents using a neural DNF module as the actor.
    """

    actor: BaseNeuralDNF

    def __init__(self, num_latent: int, use_decode_obs: bool) -> None:
        assert (
            use_decode_obs
        ), "Only decoded observation is supported for NDNF-based agent for now."
        super().__init__(num_latent, use_decode_obs)

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


class BlackjackNDNFAgent(BlackjackNDNFBasedAgent):
    """
    An agent for gymnasium Blackjack environment, with `NeuralDNF` as actor.
    This agent is not usually expected to use for training. This agent is more
    expected to be used as a post-training evaluation agent from either a
    trained `BlackjackNDNFEOAgent` or `BlackjackNDNFMutexTanhAgent`.
    To create a `BlackjackNDNFAgent` agent, pass in the following
    parameters:
    - num_latent (int): the number of conjunctions allowed in NDNF
    - use_decode_obs (bool): flag to use decode observation
    """

    actor: NeuralDNF

    def _create_default_actor(self) -> nn.Module:
        return NeuralDNF(
            self.num_inputs, self.num_latent, self.action_size, 1.0
        )


class BlackjackNDNFEOAgent(BlackjackNDNFBasedAgent):
    """
    An agent for gymnasium Blackjack environment, with `NeuralDNFEO` actor.
    This agent is used for training, and to be converted to a
    `BlackjackNDNFAgent` for post-training evaluation.
    To create a `BlackjackNDNFEOAgent` agent, pass in the following
    parameters:
    - num_latent (int): the number of conjunctions allowed in NDNF-EO
    - use_decode_obs (bool): flag to use decode observation
    """

    actor: NeuralDNFEO

    def _create_default_actor(self) -> nn.Module:
        return NeuralDNFEO(
            self.num_inputs, self.num_latent, self.action_size, 1.0
        )

    def to_ndnf_agent(self) -> BlackjackNDNFAgent:
        """
        Convert this agent to a BlackjackNDNFAgent.
        """
        ndnf_agent = BlackjackNDNFAgent(self.num_latent, self.use_decode_obs)
        ndnf_agent.actor = self.actor.to_ndnf()
        return ndnf_agent


class BlackjackNDNFMutexTanhAgent(BlackjackNDNFBasedAgent):
    """
    An agent for gymnasium Blackjack environment, with `NeuralDNFMutexTanh`
    actor.
    This agent is used for training. It can be converted to a
    `BlackjackNDNFAgent` for post-training evaluation, or used directly for
    evaluation.
    To create a `BlackjackNDNFMutexTanhAgent` agent, pass in the following
    parameters:
    - num_latent (int): the number of conjunctions allowed in the NDNF-MT
    - use_decode_obs (bool): flag to use decode observation
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
        This function is used in PPO algorithm and A2C algorithm
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
    num_latent: int,
    use_ndnf: bool,
    use_decode_obs: bool,
    use_eo: bool = False,
    use_mt: bool = False,
) -> BlackjackBaseAgent:
    if not use_ndnf:
        return BlackjackMLPAgent(num_latent, use_decode_obs)

    assert not (
        use_eo and use_mt
    ), "EO constraint and Mutex Tanh mode should not be active together."

    if not use_eo and not use_mt:
        return BlackjackNDNFAgent(num_latent, use_decode_obs)
    if use_eo and not use_mt:
        return BlackjackNDNFEOAgent(num_latent, use_decode_obs)
    return BlackjackNDNFMutexTanhAgent(num_latent, use_decode_obs)


# =============================================================================#
#                            Environment construction                          #
# =============================================================================#


def construct_single_environment(
    render_mode: str | None = "rgb_array",
) -> BlackjackEnv:
    env = gym.make("Blackjack-v1", render_mode=render_mode)
    return env  # type: ignore


def make_env(seed: int, idx: int, capture_video: bool):
    def thunk():
        if capture_video and idx == 0:
            env = construct_single_environment()
            video_dir = Path("videos")
            env = RecordVideo(env, str(video_dir.absolute()))
        else:
            env = construct_single_environment()
        env = RecordEpisodeStatistics(env)

        env.action_space.seed(seed)
        return env

    return thunk


# =============================================================================#
#                             Observation processing                           #
# =============================================================================#


def decode_tuple_obs(
    obs: gym.spaces.Tuple | tuple[int, int, int]
) -> npt.NDArray[np.float32]:
    """
    Decode the tuple into a sparse array of size 44 with 2 or 3 bit fired.
    32 bits for player sum, 11 for dealer showing, and 1 for usable ace.
    """
    out = [
        np.eye(s)[o]  # type: ignore
        for s, o in zip(BLACKJACK_TUPLE_OBS_FIRST_TWO, [obs[0], obs[1]])
    ]
    out.append(np.array([obs[2]]))
    return np.concatenate(out, dtype=np.float32)


def reverse_decode_tuple_obs(
    decoded_obs: npt.NDArray[np.float32],
) -> tuple[int, int, int]:
    """
    Reverse the decoding of the observation.
    """
    player_sum = int(np.argmax(decoded_obs[: BLACKJACK_TUPLE_OBS_FIRST_TWO[0]]))
    dealer_showing = int(
        np.argmax(
            decoded_obs[
                BLACKJACK_TUPLE_OBS_FIRST_TWO[0] : sum(
                    BLACKJACK_TUPLE_OBS_FIRST_TWO
                )
            ]
        )
    )
    return (player_sum, dealer_showing, int(decoded_obs[-1]))


def non_decode_obs(
    obs: gym.spaces.Tuple | tuple[int, int, int], normalise: bool
) -> npt.NDArray[np.float32]:
    """
    Keep the observation as tuple and normalise if needed.
    """
    if normalise:
        return np.array(
            [
                o / s  # type: ignore
                for s, o in zip(BLACKJACK_TUPLE_OBS_FIRST_TWO, [obs[0], obs[1]])
            ]
            + [obs[2]],
            dtype=np.float32,
        )
    return np.array(obs, dtype=np.float32)


def blackjack_env_preprocess_obss(
    obs_tuple: tuple[npt.NDArray],
    use_ndnf: bool,
    device: torch.device,
    normalise: bool = False,
) -> dict[str, Tensor]:
    tuple_array = np.stack(obs_tuple, axis=1)

    input_np_arr = np.stack([non_decode_obs(t, normalise) for t in tuple_array])
    decode_input_nd_array = np.stack([decode_tuple_obs(t) for t in tuple_array])

    if use_ndnf:
        decode_input_nd_array = np.where(
            decode_input_nd_array == 0, -1, decode_input_nd_array
        )

    return {
        "input": torch.tensor(
            input_np_arr,
            dtype=torch.float32,
            device=device,
        ),
        "decode_input": torch.tensor(
            decode_input_nd_array,
            dtype=torch.float32,
            device=device,
        ),
    }


# =============================================================================#
#                         Policy grid & comparison                             #
# =============================================================================#


def get_target_policy(csv_path: Path) -> TargetPolicyType:
    """
    From the csv file, get the target policy learned by a tabular Q-learning
    agent. The return dict is sorted by the observation tuple with the action as
    its value.
    """
    d = pd.read_csv(csv_path, header=[0, 1, 2], dtype=float).to_dict()
    d.pop(("Unnamed: 0_level_0", "Unnamed: 0_level_1", "Unnamed: 0_level_2"))

    unsorted_target_q_value_table = {
        tuple(map(int, k)): v for k, v in d.items()  # type: ignore
    }
    keys: list[tuple[int, int, int]] = sorted(
        [tuple(map(int, k)) for k in d.keys()]  # type: ignore
    )
    target_q_value_table: TargetQValueTableType = OrderedDict(
        [(k, unsorted_target_q_value_table[k]) for k in keys]
    )
    return OrderedDict(
        [
            (obs, int(np.argmax([q_vals[k] for k in sorted(q_vals.keys())])))
            for obs, q_vals in target_q_value_table.items()
        ]
    )


def _create_grid(policy: dict) -> tuple[np.ndarray, np.ndarray]:
    player_count, dealer_count = np.meshgrid(
        # players count, dealers face-up card
        np.arange(12, 22),
        np.arange(1, 11),
    )
    policy_grid_useable_ace = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], 1)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    policy_grid_no_useable_ace = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], 0)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    return policy_grid_useable_ace, policy_grid_no_useable_ace


def _create_model_grids(
    target_policy: TargetPolicyType,
    model_action_distribution: Any,
    argmax: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model_policy = defaultdict(int)
    for i, obs in enumerate(target_policy.keys()):
        if argmax:
            a = model_action_distribution[i].argmax().item()
        else:
            # Take the probability of taking the action 'HIT' (1)
            a = model_action_distribution[i][1].item()
        model_policy[obs] = a
    return _create_grid(model_policy)


def _generate_policy_with_diff_support(
    policy_ace: np.ndarray,
    policy_no_ace: np.ndarray,
    target_policy_ace: np.ndarray | None = None,
    target_policy_no_ace: np.ndarray | None = None,
    suptitle: str | None = None,
    argmax: bool = True,
    plot_diff: bool = False,
):
    fig = plt.figure(figsize=figaspect(0.4))
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)

    # Plot the policy with usable ace
    fig.add_subplot(1, 2, 1)
    ax1 = sns.heatmap(
        policy_ace,
        linewidth=0,
        annot=True,
        cmap="Accent_r",
        cbar=False,
        fmt=".2f" if not argmax else "d",
    )
    if argmax and plot_diff:
        assert target_policy_ace is not None
        ax1 = sns.heatmap(
            policy_ace,
            mask=target_policy_ace == policy_ace,
            annot=True,
            cmap="Blues",
            cbar=False,
        )
    ax1.set_title("Policy with usable ace")
    ax1.set_xlabel("Player sum")
    ax1.set_ylabel("Dealer showing")
    ax1.set_xticklabels(range(12, 22))  # type: ignore
    ax1.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)  # type: ignore

    fig.add_subplot(1, 2, 2)
    ax2 = sns.heatmap(
        policy_no_ace,
        linewidth=0,
        annot=True,
        cmap="Accent_r",
        cbar=False,
        fmt=".2f" if not argmax else "d",
    )
    if argmax and plot_diff:
        assert target_policy_no_ace is not None
        ax2 = sns.heatmap(
            policy_no_ace,
            mask=policy_no_ace == target_policy_no_ace,
            annot=True,
            cmap="Blues",
            cbar=False,
        )
    ax2.set_title("Policy without usable ace")
    ax2.set_xlabel("Player sum")
    ax2.set_ylabel("Dealer showing")
    ax2.set_xticklabels(range(12, 22))  # type: ignore
    ax2.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)  # type: ignore

    # add a legend
    if argmax:
        legend_elements = [
            Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
            Patch(facecolor="grey", edgecolor="black", label="Stick"),
        ]
        ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))

    return fig


def create_target_policy_plots(
    target_policy: TargetPolicyType, model_name: str
) -> Figure:
    target_policy_ace, target_policy_no_ace = _create_grid(target_policy)
    policy_type = "Argmax Policy"
    return _generate_policy_with_diff_support(
        policy_ace=target_policy_ace,
        policy_no_ace=target_policy_no_ace,
        suptitle=f"{policy_type} for {model_name}",
        argmax=True,
    )


def create_policy_plots_from_action_distribution(
    target_policy: TargetPolicyType,
    model_action_distribution: Any,
    model_name: str,
    argmax: bool = True,
    plot_diff: bool = False,
) -> Figure:
    """
    model_action_distribution: should be `.prob` output of a distribution
    """
    policy_ace, policy_no_ace = _create_model_grids(
        target_policy, model_action_distribution, argmax=argmax
    )
    target_policy_ace, target_policy_no_ace = _create_grid(target_policy)
    policy_type = "Argmax Policy" if argmax else "Soft Policy (HIT)"
    return _generate_policy_with_diff_support(
        policy_ace=policy_ace,
        policy_no_ace=policy_no_ace,
        target_policy_ace=target_policy_ace,
        target_policy_no_ace=target_policy_no_ace,
        suptitle=f"{policy_type} for {model_name}",
        argmax=argmax,
        plot_diff=plot_diff,
    )


def create_policy_plots_from_asp(
    target_policy: TargetPolicyType,
    asp_policy: TargetPolicyType,
    model_name: str,
    argmax: bool = True,
    plot_diff: bool = False,
):
    target_policy_ace, target_policy_no_ace = _create_grid(target_policy)
    asp_policy_ace, asp_policy_no_ace = _create_grid(asp_policy)
    policy_type = "Argmax Policy" if argmax else "Soft Policy (HIT)"
    return _generate_policy_with_diff_support(
        policy_ace=asp_policy_ace,
        policy_no_ace=asp_policy_no_ace,
        target_policy_ace=target_policy_ace,
        target_policy_no_ace=target_policy_no_ace,
        suptitle=f"{policy_type} for {model_name}",
        argmax=argmax,
        plot_diff=plot_diff,
    )
