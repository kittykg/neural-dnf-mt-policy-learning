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

from taxi_common import (
    TaxiEnvPPOBaseAgent,
    TaxiEnvPPONDNFBasedAgent,
    make_env,
    taxi_env_preprocess_obs,
)


DEFAULT_GEN_SEED = 3
ENV_NUM_EPISODES = 10000
EVAL_NUM_ENVS = 8
N_ACTIONS = 6


def eval_model_on_environment(
    model: TaxiEnvPPOBaseAgent,
    device: torch.device,
    use_argmax: bool = True,
    eval_num_runs: int = ENV_NUM_EPISODES,
) -> dict[str, Any]:
    model.to(device)
    use_ndnf = isinstance(model, TaxiEnvPPONDNFBasedAgent)

    logs: dict[str, Any] = {
        "num_frames_per_episode": [],
        "return_per_episode": [],
    }
    if use_ndnf and use_argmax:
        logs["mutual_exclusivity"] = True
        logs["missing_actions"] = False

    envs = gym.vector.SyncVectorEnv(
        [make_env(i, i, False) for i in range(EVAL_NUM_ENVS)],
    )

    next_obs, _ = envs.reset()
    next_obs_dict = taxi_env_preprocess_obs(next_obs, use_ndnf, device)

    log_done_counter = 0
    log_episode_return = torch.zeros(EVAL_NUM_ENVS, device=device)
    log_episode_num_frames = torch.zeros(EVAL_NUM_ENVS, device=device)

    while log_done_counter < eval_num_runs:
        if use_ndnf:
            # For NDNF based model, the get_actions() returns a tuple of
            # actions and tanh interpretation. We check the if the tanh
            # interpretation is greater than 0.
            with torch.no_grad():
                actions = model.get_actions(
                    preprocessed_obs=next_obs_dict, use_argmax=use_argmax
                )
            if use_argmax:
                tanh_action = np.count_nonzero(actions[1] > 0, axis=1)
                if np.any(tanh_action > 1):
                    logs["mutual_exclusivity"] = False
                if np.any(tanh_action == 0):
                    logs["missing_actions"] = True
            actions = actions[0]
        else:
            # MLP based model
            with torch.no_grad():
                actions = model.get_actions(
                    preprocessed_obs=next_obs_dict, use_argmax=use_argmax
                )

        next_obs, reward, terminations, truncations, _ = envs.step(actions)
        next_obs_dict = taxi_env_preprocess_obs(next_obs, use_ndnf, device)
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

    envs.close()

    return logs


def eval_model_on_all_possible_states():
    # TODO: Implement this function
    pass
