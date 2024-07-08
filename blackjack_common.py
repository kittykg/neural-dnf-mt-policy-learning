from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any


import gymnasium as gym
from matplotlib.figure import Figure, figaspect
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import pandas as pd
import seaborn as sns


BLACKJACK_TUPLE_OBS_FIRST_TWO = [32, 11]
N_ACTIONS = 2
N_OBSERVATION_DECODE_SIZE = 44  # 32 + 11 + 1
N_OBSERVATION_SIZE = 3


TargetQValueTableType = OrderedDict[tuple[int, int, int], dict[int, float]]
TargetPolicyType = OrderedDict[tuple[int, int, int], int]


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
