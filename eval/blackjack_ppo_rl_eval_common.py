from pathlib import Path
import sys
from typing import Any

import gymnasium as gym
import numpy as np
import torch

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from common import DiversityScoreTracker
from blackjack_common import *


EVAL_NUM_RUNS = 1000000
EVAL_NUM_ENVS = 16


def get_relevant_targets_from_target_policy(
    use_ndnf: bool,
    target_policy: TargetPolicyType,
    device: torch.device,
    normalise: bool = False,
) -> dict[str, Any]:
    """
    Return the observations, preprocessed observations and target actions from a
    target policy trained from tabular Q-learning.
    """
    obs_list = [obs for obs in target_policy.keys()]
    target_q_actions = np.array([target_policy[obs] for obs in obs_list])

    input_np_arr = np.stack(
        [non_decode_obs(obs, normalise) for obs in obs_list]
    )
    decode_input_nd_array = np.stack(
        [decode_tuple_obs(obs) for obs in obs_list]
    )

    if use_ndnf:
        decode_input_nd_array = np.where(
            decode_input_nd_array == 0, -1, decode_input_nd_array
        )

    obs_dict = {
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

    return {
        "obs_list": obs_list,
        "obs_dict": obs_dict,
        "target_q_actions": target_q_actions,
    }


def get_agent_policy(
    agent: BlackjackBaseAgent,
    target_q_policy: TargetPolicyType,
    device: torch.device,
    normalise: bool = False,
) -> Any:
    """
    Return the action distribution for the agent at all states presented in
    `target_q_policy`.
    """
    obs_dict = get_relevant_targets_from_target_policy(
        isinstance(agent, BlackjackNDNFBasedAgent),
        target_q_policy,
        device,
        normalise,
    )["obs_dict"]

    with torch.no_grad():
        action_dist = agent.get_action_distribution(obs_dict)

    return action_dist.probs.cpu().numpy()  # type: ignore


def mlp_agent_cmp_target_csv(
    target_policy_csv_path: Path,
    agent: BlackjackMLPAgent,
    device: torch.device,
    normalise: bool = False,
) -> dict[str, Any]:
    logs: dict[str, Any] = {}

    target_policy = get_target_policy(target_policy_csv_path)
    ret = get_relevant_targets_from_target_policy(
        False, target_policy, device, normalise
    )
    obs_dict = ret["obs_dict"]
    target_q_actions = ret["target_q_actions"]

    dst = DiversityScoreTracker(N_ACTIONS)
    with torch.no_grad():
        actions = agent.get_actions(obs_dict)
        dst.update(actions)

    policy_error_cmp_to_q = np.count_nonzero(actions != target_q_actions) / len(
        target_q_actions
    )
    logs["policy_error_cmp_to_q"] = policy_error_cmp_to_q
    logs["action_diversity_score"] = dst.compute_diversity_score()
    logs["action_entropy"] = dst.compute_entropy()

    return logs


def ndnf_based_agent_cmp_target_csv(
    target_policy_csv_path: Path,
    agent: BlackjackNDNFBasedAgent,
    device: torch.device,
    normal_indices: list[int] | None = None,
) -> dict[str, Any]:
    logs: dict[str, Any] = {
        "mutual_exclusivity": True,
        "missing_actions": False,
    }

    target_policy = get_target_policy(target_policy_csv_path)
    ret = get_relevant_targets_from_target_policy(True, target_policy, device)
    obs_list = ret["obs_list"]
    if normal_indices is not None:
        obs_list = [obs_list[i] for i in normal_indices]
    obs_dict = ret["obs_dict"]
    if normal_indices is not None:
        obs_dict = {k: v[normal_indices] for k, v in obs_dict.items()}
    target_q_actions = ret["target_q_actions"]
    if normal_indices is not None:
        target_q_actions = target_q_actions[normal_indices]

    dst = DiversityScoreTracker(N_ACTIONS)
    with torch.no_grad():
        actions, tanh_actions = agent.get_actions(
            preprocessed_obs=obs_dict, use_argmax=True
        )
        logs["actions"] = actions
        dst.update(actions)

    tanh_actions_discretised = np.count_nonzero(tanh_actions > 0, axis=1)
    me_violation_indices = None
    ma_indices = None

    # Check for mutual exclusivity violations
    if np.any(tanh_actions_discretised > 1):
        logs["mutual_exclusivity"] = False
        logs["mutual_exclusivity_violations_count"] = int(
            np.count_nonzero(tanh_actions_discretised > 1)
        )
        me_violation_indices = np.where(tanh_actions_discretised > 1)[
            0
        ].tolist()
        logs["mutual_exclusivity_violations_indices"] = me_violation_indices
        logs["mutual_exclusivity_violations_states"] = [
            obs_list[i] for i in me_violation_indices
        ]

    # Check for missing actions
    if np.any(tanh_actions_discretised == 0):
        logs["missing_actions"] = True
        logs["missing_actions_count"] = int(
            np.count_nonzero(tanh_actions_discretised == 0)
        )
        ma_indices = np.where(tanh_actions_discretised == 0)[0].tolist()
        logs["missing_actions_indices"] = ma_indices
        logs["missing_actions_states"] = [obs_list[i] for i in ma_indices]

    # Compute indices
    indices_separation_dict = (
        get_abnormal_and_normal_state_indices_from_ndnf_based_agent(
            obs_list, me_violation_indices, ma_indices
        )
    )
    logs.update(indices_separation_dict)

    # Compute other metrics
    policy_error_cmp_to_q = np.count_nonzero(actions != target_q_actions) / len(
        target_q_actions
    )

    logs["policy_error_cmp_to_q"] = policy_error_cmp_to_q
    logs["action_diversity_score"] = dst.compute_diversity_score()
    logs["action_entropy"] = dst.compute_entropy()

    return logs


def get_abnormal_and_normal_state_indices_from_ndnf_based_agent(
    all_obs_list: list,
    me_violation_indices: list[int] | None,
    ma_indices: list[int] | None,
) -> dict[str, list]:
    ABNORMAL_INDICES_KEYS = "abnormal_indices"
    NORMAL_INDICES_KEYS = "normal_indices"

    all_possible_idx = set(range(len(all_obs_list)))
    if me_violation_indices is None and ma_indices is None:
        return {
            ABNORMAL_INDICES_KEYS: [],
            NORMAL_INDICES_KEYS: list(all_possible_idx),
        }

    if me_violation_indices is None and ma_indices is not None:
        return {
            ABNORMAL_INDICES_KEYS: ma_indices,
            NORMAL_INDICES_KEYS: sorted(
                list(all_possible_idx.difference(ma_indices))
            ),
        }

    if me_violation_indices is not None and ma_indices is None:
        return {
            ABNORMAL_INDICES_KEYS: me_violation_indices,
            NORMAL_INDICES_KEYS: sorted(
                list(all_possible_idx.difference(me_violation_indices))
            ),
        }

    assert me_violation_indices is not None and ma_indices is not None
    overall_idx_set = set(me_violation_indices).union(ma_indices)
    normal_idx_list = sorted(list(all_possible_idx.difference(overall_idx_set)))

    return {
        ABNORMAL_INDICES_KEYS: sorted(list(overall_idx_set)),
        NORMAL_INDICES_KEYS: normal_idx_list,
    }


def eval_on_environments(
    model: BlackjackBaseAgent,
    device: torch.device,
    use_argmax: bool = True,
    eval_num_runs: int = EVAL_NUM_RUNS,
) -> dict[str, list[float]]:
    """
    Evaluate the model over a number of episodes across the parallel
    environments, also return its action distribution
    """
    envs = gym.vector.SyncVectorEnv(
        [make_env(i, i, False) for i in range(EVAL_NUM_ENVS)]
    )
    use_ndnf = isinstance(model, BlackjackNDNFBasedAgent)

    logs: dict[str, Any] = {
        "num_frames_per_episode": [],
        "return_per_episode": [],
    }

    next_obs, _ = envs.reset()
    next_obs_dict = blackjack_env_preprocess_obss(next_obs, use_ndnf, device)

    log_done_counter = 0
    log_episode_return = torch.zeros(EVAL_NUM_ENVS, device=device)
    log_episode_num_frames = torch.zeros(EVAL_NUM_ENVS, device=device)

    while log_done_counter < eval_num_runs:
        with torch.no_grad():
            actions = model.get_actions(
                preprocessed_obs=next_obs_dict, use_argmax=use_argmax
            )
            if use_ndnf:
                # For NDNF based AC model, the get_actions() returns a tuple of
                # actions and tanh interpretation. In this evaluation, we only
                # take the actions and ignore the tanh interpretation.
                actions = actions[0]
        next_obs, reward, terminations, truncations, _ = envs.step(actions)
        next_obs_dict = blackjack_env_preprocess_obss(
            next_obs, use_ndnf, device
        )
        next_done = np.logical_or(terminations, truncations)

        log_episode_return += torch.tensor(
            reward, device=device, dtype=torch.float
        )
        log_episode_num_frames += torch.ones(EVAL_NUM_ENVS, device=device)

        for i, done in enumerate(next_done):
            if done:
                log_done_counter += 1
                logs["return_per_episode"].append(log_episode_return[i].item())
                logs["num_frames_per_episode"].append(
                    log_episode_num_frames[i].item()
                )

        mask = 1 - torch.tensor(next_done, device=device, dtype=torch.float)
        log_episode_return *= mask
        log_episode_num_frames *= mask

    # Calculate win rate
    num_wins = np.sum(np.array(logs["return_per_episode"]) == 1)
    num_losses = np.sum(np.array(logs["return_per_episode"]) == -1)
    num_draws = np.sum(np.array(logs["return_per_episode"]) == 0)

    logs["num_wins"] = num_wins
    logs["num_losses"] = num_losses
    logs["num_draws"] = num_draws
    logs["win_rate"] = num_wins / eval_num_runs
    logs["avg_return_per_episode"] = np.mean(logs["return_per_episode"])

    envs.close()

    return logs
