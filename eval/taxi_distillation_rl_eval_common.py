from enum import Enum
from pathlib import Path
import sys
from typing import Any

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import numpy as np
import torch
from torch import Tensor
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from neural_dnf.neural_dnf import (
    BaseNeuralDNF,
    NeuralDNFEO,
    BaseNeuralDNFMutexTanh,
)

from taxi_common import (
    N_OBSERVATION_SIZE,
    split_all_states_to_reachable_and_non,
    taxi_env_preprocess_obs,
)


EVAL_NUM_ENVS = 8
EVAL_NUM_RUNS = 1000000
TAXI_ENV_POSSIBLE_STATES, _ = split_all_states_to_reachable_and_non()


class EnvEvalLogKeys(Enum):
    ME = "env_eval_mutual_exclusivity"
    MA = "env_eval_missing_actions"
    HAS_TRUNC = "env_eval_has_truncation"
    AVG_L1_MOD_AUX = "env_eval_avg_l1_mod_aux"
    AVG_RETURN_PER_EPISODE = "env_eval_avg_return_per_episode"
    AVG_NUM_FRAMES_PER_EPISODE = "env_eval_avg_num_frames_per_episode"


class StateEvalLogKeys(Enum):
    ME = "state_eval_mutual_exclusivity"
    ME_COUNT = "states_eval_mutual_exclusivity_violations_count"
    ME_INDICES = "states_eval_mutual_exclusivity_violations_indices"
    ME_STATES = "states_eval_mutual_exclusivity_violations_states"
    MA = "state_eval_missing_actions"
    MA_COUNT = "states_eval_missing_actions_count"
    MA_INDICES = "states_eval_missing_actions_indices"
    MA_STATES = "states_eval_missing_actions_states"
    ACTIONS = "state_eval_states_model_actions"
    TANH_OUT = "state_eval_states_model_tanh_out"
    ACTION_DISTRIBUTION = "state_eval_action_distribution"
    KL_DIV = "state_eval_kl_div"
    POLICY_ERROR_CMP_TARGET = "state_eval_policy_error_cmp_target"
    POLICY_ERROR_RATE_CMP_TARGET = "state_eval_policy_error_rate_cmp_target"


def eval_get_ndnf_action(
    ndnf_model: BaseNeuralDNF,
    obs: np.ndarray,
    device: torch.device,
    use_argmax: bool = True,
) -> tuple[Tensor, Tensor]:
    preprocessed_obs = taxi_env_preprocess_obs(
        obs, use_ndnf=True, device=device
    )
    if ndnf_model.conjunctions.weights.data.shape[1] == N_OBSERVATION_SIZE:
        obs_key = "input"
    else:
        obs_key = "decode_input"
    x = preprocessed_obs[obs_key]

    with torch.no_grad():
        if isinstance(ndnf_model, BaseNeuralDNFMutexTanh):
            all_form_dict = ndnf_model.get_all_forms(x)
            action_prob = (all_form_dict["disjunction"]["mutex_tanh"] + 1) / 2
            if use_argmax:
                action = torch.argmax(action_prob, dim=1)
            else:
                action = Categorical(probs=action_prob).sample()
            tanh_out = all_form_dict["disjunction"]["tanh"]
        else:
            out = ndnf_model(x)
            action = torch.argmax(out, dim=1)
            tanh_out = torch.tanh(out)

    return action, tanh_out


def eval_get_ndnf_mt_action_dist(
    ndnf_model: BaseNeuralDNFMutexTanh, obs: np.ndarray, device: torch.device
) -> Categorical:
    preprocessed_obs = taxi_env_preprocess_obs(
        obs, use_ndnf=True, device=device
    )
    if ndnf_model.conjunctions.weights.data.shape[1] == N_OBSERVATION_SIZE:
        obs_key = "input"
    else:
        obs_key = "decode_input"
    x = preprocessed_obs[obs_key]

    with torch.no_grad():
        action_dist = (ndnf_model(x) + 1) / 2
    return Categorical(probs=action_dist)


def eval_on_environments(
    ndnf_model: BaseNeuralDNF,
    device: torch.device,
    use_argmax: bool = True,
    num_episodes: int = EVAL_NUM_RUNS,
) -> dict[str, Any]:
    # Convert NDNF-EO to plain NDNF
    if isinstance(ndnf_model, NeuralDNFEO):
        eval_model = ndnf_model.to_ndnf()
    else:
        eval_model = ndnf_model

    eval_model.to(device)
    eval_model.eval()

    num_frames_per_episode = []
    return_per_episode = []

    logs: dict[str, Any] = {
        EnvEvalLogKeys.ME.value: True,
        EnvEvalLogKeys.MA.value: False,
        EnvEvalLogKeys.HAS_TRUNC.value: False,
    }

    envs = SyncVectorEnv(
        [
            lambda: gym.make("Taxi-v3", render_mode="rgb_array")
            for _ in range(EVAL_NUM_ENVS)
        ]
    )

    obs, _ = envs.reset()

    log_done_counter = 0
    log_episode_return = torch.zeros(EVAL_NUM_ENVS, device=device)
    log_episode_num_frames = torch.zeros(EVAL_NUM_ENVS, device=device)

    while log_done_counter < num_episodes:
        with torch.no_grad():
            actions, tanh_out = eval_get_ndnf_action(
                eval_model, obs, device, use_argmax
            )

        tanh_action = np.count_nonzero(tanh_out > 0, axis=1)
        if np.any(tanh_action > 1):
            logs[EnvEvalLogKeys.ME.value] = False
        if np.any(tanh_action == 0):
            logs[EnvEvalLogKeys.MA.value] = True

        obs, reward, terminated, truncated, _ = envs.step(actions.cpu().numpy())
        if np.any(truncated):
            logs[EnvEvalLogKeys.HAS_TRUNC.value] = True
        done = tuple(a | b for a, b in zip(terminated, truncated))  # type: ignore

        log_episode_return += torch.tensor(
            reward, device=device, dtype=torch.float
        )
        log_episode_num_frames += torch.ones(EVAL_NUM_ENVS, device=device)

        for i, d in enumerate(done):
            if d:
                log_done_counter += 1
                return_per_episode.append(log_episode_return[i].item())
                num_frames_per_episode.append(log_episode_num_frames[i].item())

        mask = 1 - torch.tensor(done, device=device, dtype=torch.float)
        log_episode_return *= mask
        log_episode_num_frames *= mask

    envs.close()

    # Calculate the sparsity of the model
    p_t = torch.cat(
        [
            parameter.view(-1)
            for parameter in eval_model.parameters()
            if parameter.requires_grad
        ]
    )

    logs[EnvEvalLogKeys.AVG_L1_MOD_AUX.value] = (
        torch.abs(p_t * (6 - torch.abs(p_t))).sum() / p_t.numel()
    ).item()
    logs[EnvEvalLogKeys.AVG_RETURN_PER_EPISODE.value] = float(
        np.mean(return_per_episode)
    )
    logs[EnvEvalLogKeys.AVG_NUM_FRAMES_PER_EPISODE.value] = float(
        np.mean(num_frames_per_episode)
    )

    return logs


def eval_on_all_possible_states(
    ndnf_model: BaseNeuralDNF,
    device: torch.device,
    target_q_table: np.ndarray | None = None,
    target_action_dist: Categorical | None = None,
):
    # By default, if target_q_table is provided, we prioritise using it with
    # argmax action
    # If target_action_dist is provided, we can choose whether compute the
    # policy error with argmax action and compute the KL divergence between the
    # model's action distribution and the target action distribution
    assert (
        target_q_table is not None or target_action_dist is not None
    ), "Either target_q_table or target_action_dist must be provided"

    # Convert NDNF-EO to plain NDNF
    if isinstance(ndnf_model, NeuralDNFEO):
        eval_model = ndnf_model.to_ndnf()
    else:
        eval_model = ndnf_model

    eval_model.to(device)
    eval_model.eval()

    logs: dict[str, Any] = {
        StateEvalLogKeys.ME.value: True,
        StateEvalLogKeys.MA.value: False,
    }

    with torch.no_grad():
        actions, tanh_out = eval_get_ndnf_action(
            eval_model, np.array(TAXI_ENV_POSSIBLE_STATES), device
        )
        logs[StateEvalLogKeys.ACTIONS.value] = actions
        logs[StateEvalLogKeys.TANH_OUT.value] = tanh_out

        if isinstance(eval_model, BaseNeuralDNFMutexTanh):
            action_dist = eval_get_ndnf_mt_action_dist(
                eval_model, np.array(TAXI_ENV_POSSIBLE_STATES), device
            )
            logs[StateEvalLogKeys.ACTION_DISTRIBUTION.value] = action_dist.probs  # type: ignore

    tanh_actions_discretised = np.count_nonzero(tanh_out > 0, axis=1)

    me_violation_indices = None
    ma_indices = None

    # Check for mutual exclusivity violations
    if np.any(tanh_actions_discretised > 1):
        logs[StateEvalLogKeys.ME.value] = False
        logs[StateEvalLogKeys.ME_COUNT.value] = int(
            np.count_nonzero(tanh_actions_discretised > 1)
        )
        me_violation_indices = np.where(tanh_actions_discretised > 1)[
            0
        ].tolist()
        logs[StateEvalLogKeys.ME_INDICES.value] = me_violation_indices
        logs[StateEvalLogKeys.ME_STATES.value] = [
            TAXI_ENV_POSSIBLE_STATES[i] for i in me_violation_indices
        ]

    # Check for missing actions
    if np.any(tanh_actions_discretised == 0):
        logs[StateEvalLogKeys.MA.value] = True
        logs[StateEvalLogKeys.MA_COUNT.value] = int(
            np.count_nonzero(tanh_actions_discretised == 0)
        )
        ma_indices = np.where(tanh_actions_discretised == 0)[0].tolist()
        logs[StateEvalLogKeys.MA_INDICES.value] = ma_indices
        logs[StateEvalLogKeys.MA_STATES.value] = [
            TAXI_ENV_POSSIBLE_STATES[i] for i in ma_indices
        ]

    # Calculate the policy error
    if target_q_table is not None:
        target_q_argmax_actions = np.argmax(target_q_table, axis=1)
        policy_error = np.where(actions.numpy() != target_q_argmax_actions)[0]
        policy_error_rate = np.count_nonzero(
            actions.numpy() != target_q_argmax_actions
        ) / len(actions)
    else:
        # target_action_dist is not None
        assert target_action_dist is not None
        target_action_dist_probs = target_action_dist.probs.numpy()  # type: ignore
        target_action_dist_argmax = np.argmax(target_action_dist_probs, axis=1)
        policy_error = np.where(actions.numpy() != target_action_dist_argmax)[0]
        policy_error_rate = np.count_nonzero(
            actions.numpy() != target_action_dist_argmax
        ) / len(actions)
        if isinstance(eval_model, BaseNeuralDNFMutexTanh):
            kl_div = F.kl_div(
                input=torch.log(action_dist.probs + 1e-8),  # type: ignore
                target=target_action_dist.probs,  # type: ignore
                reduction="batchmean",
            )
            logs[StateEvalLogKeys.KL_DIV.value] = kl_div.item()

    logs[StateEvalLogKeys.POLICY_ERROR_CMP_TARGET.value] = policy_error
    logs[StateEvalLogKeys.POLICY_ERROR_RATE_CMP_TARGET.value] = (
        policy_error_rate
    )

    return logs
