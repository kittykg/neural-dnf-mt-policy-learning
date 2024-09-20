from enum import Enum
import json
from pathlib import Path
import sys
from typing import Any

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import numpy as np
from omegaconf import DictConfig, OmegaConf
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
from taxi_distillation_common import (
    load_mlp_model,
    generate_data_from_mlp,
    load_target_q_table,
)


EVAL_NUM_ENVS = 8
EVAL_NUM_RUNS = 1000000
TAXI_ENV_POSSIBLE_STATES, _ = split_all_states_to_reachable_and_non()
RESULT_STORAGE_DIR = root / "results" / "Taxi-Distillation"
START_STATE_SEEDS_JSON_NAME = "taxi_all_start_states.json"


class EnvEvalLogKeys(Enum):
    ME = "env_eval_mutual_exclusivity"
    MA = "env_eval_missing_actions"
    HAS_TRUNC = "env_eval_has_truncation"
    AVG_L1_MOD_AUX = "env_eval_avg_l1_mod_aux"
    AVG_RETURN_PER_EPISODE = "env_eval_avg_return_per_episode"
    AVG_NUM_FRAMES_PER_EPISODE = "env_eval_avg_num_frames_per_episode"
    TRUNC_SEEDS = "env_eval_truncation_seeds"


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


def eval_on_environments_with_all_start_seeds(
    ndnf_model: BaseNeuralDNF,
    device: torch.device,
    all_seeds_for_each_start_state: list[int],
    use_argmax: bool = True,
):
    """
    `all_seeds_for_each_start_state` is expected to be a list of 300 integers
    """
    # Convert NDNF-EO to plain NDNF
    if isinstance(ndnf_model, NeuralDNFEO):
        eval_model = ndnf_model.to_ndnf()
    else:
        eval_model = ndnf_model

    num_frames_per_episode = []
    return_per_episode = []

    logs: dict[str, Any] = {
        EnvEvalLogKeys.ME.value: True,
        EnvEvalLogKeys.MA.value: False,
        EnvEvalLogKeys.HAS_TRUNC.value: False,
    }

    num_envs = 10
    envs = SyncVectorEnv(
        [
            lambda: gym.make("Taxi-v3", render_mode="rgb_array")
            for _ in range(num_envs)
        ]
    )

    seed_batch = np.array(all_seeds_for_each_start_state).reshape(-1, num_envs)
    return_per_episode = np.zeros(seed_batch.shape)
    num_frames_per_episode = np.zeros(seed_batch.shape)
    trunc_seed = []

    for batch_idx in range(seed_batch.shape[0]):
        sb = seed_batch[batch_idx].tolist()
        obs, _ = envs.reset(seed=sb)

        log_episode_return = torch.zeros(num_envs, device=device)
        log_episode_num_frames = torch.zeros(num_envs, device=device)
        log_done_counter = np.zeros(num_envs).astype(bool)

        while not np.all(log_done_counter):
            with torch.no_grad():
                actions, tanh_out = eval_get_ndnf_action(
                    eval_model, obs, device, use_argmax
                )

            tanh_action = np.count_nonzero(tanh_out > 0, axis=1)
            if np.any(tanh_action > 1):
                logs[EnvEvalLogKeys.ME.value] = False
            if np.any(tanh_action == 0):
                logs[EnvEvalLogKeys.MA.value] = True

            obs, reward, terminated, truncated, _ = envs.step(
                actions.cpu().numpy()
            )
            if np.any(truncated):
                logs[EnvEvalLogKeys.HAS_TRUNC.value] = True

            log_episode_return += torch.tensor(
                reward, device=device, dtype=torch.float
            )
            log_episode_num_frames += torch.ones(num_envs, device=device)

            for i, (ter, trun) in enumerate(zip(terminated, truncated)):
                if ter or trun:
                    log_done_counter[i] = True
                    return_per_episode[batch_idx, i] = log_episode_return[
                        i
                    ].item()
                if trun:
                    trunc_seed.append(sb[i])

            mask = 1 - torch.tensor(
                tuple(a | b for a, b in zip(terminated, truncated)),
                device=device,
                dtype=torch.float,
            )
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
    logs[EnvEvalLogKeys.TRUNC_SEEDS.value] = trunc_seed

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


def get_target_q_table_and_action_dist(
    eval_cfg: DictConfig, device: torch.device
) -> tuple[np.ndarray | None, Categorical | None]:
    target_q_table = None
    target_action_dist = None

    if eval_cfg["distillation_mlp"]["mlp_model_path"] is not None:
        # Pass a dummy config to load_model
        distillation_mlp_cfg: dict[str, Any] = OmegaConf.to_container(
            eval_cfg["distillation_mlp"].copy()
        )  # type: ignore
        mlp_model_path_str = distillation_mlp_cfg.pop("mlp_model_path")
        mlp_model = load_mlp_model(
            model_architecture_cfg=distillation_mlp_cfg,
            mlp_model_path_str=mlp_model_path_str,
            device=device,
        )
        _, target_action_dist = generate_data_from_mlp(mlp_model, device)

    else:
        assert (
            eval_cfg["distillation_tab_q"]["tab_q_path"] is not None
        ), "Either mlp_model_path or tab_q_path must be provided"

        tab_q_path_str = eval_cfg["distillation_tab_q"]["tab_q_path"]
        target_q_table = load_target_q_table(tab_q_path_str)

    return target_q_table, target_action_dist


def generate_all_possible_start_states() -> list[int]:
    start_state_counter = {}
    env = gym.make("Taxi-v3")
    for i in range(1000000):
        obs, _ = env.reset(seed=i)

        if obs not in start_state_counter:
            start_state_counter[obs] = []
        start_state_counter[obs].append(i)

    sorted_k = sorted(list(start_state_counter.keys()))

    json_dict = {
        "all_possible_start_states": sorted_k,
        "num_possible_start_states": len(sorted_k),
        "all_seeds_for_each_start_state": [
            min(start_state_counter[k]) for k in sorted_k
        ],
        "state_seed_map": {k: min(start_state_counter[k]) for k in sorted_k},
    }

    with open(RESULT_STORAGE_DIR / START_STATE_SEEDS_JSON_NAME, "w") as f:
        json.dump(json_dict, f, indent=4)

    return json_dict["all_seeds_for_each_start_state"]


def get_all_possible_seeds_for_all_start_states() -> list[int]:
    if not (RESULT_STORAGE_DIR / START_STATE_SEEDS_JSON_NAME).exists():
        return generate_all_possible_start_states()

    with open(RESULT_STORAGE_DIR / START_STATE_SEEDS_JSON_NAME, "r") as f:
        json_dict = json.load(f)
        return json_dict["all_seeds_for_each_start_state"]
