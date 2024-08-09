# This script evaluates the NDNF based agent on POMDP SpecialStateCorridor envs
# i.e. using wall status as observation of agent
from copy import deepcopy
import json
import logging
from pathlib import Path
import random
import sys
import traceback
from typing import Any, Callable

import gymnasium as gym
import hydra
import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig
import torch
from torch import Tensor


file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass


from ss_corridor_ppo import (
    construct_model,
    construct_single_environment,
    make_env,
    ss_corridor_preprocess_obs,
    SSCPPONDNFMutexTanhAgent,
)
from utils import post_to_discord_webhook


DEFAULT_GEN_SEED = 2
NUM_PROCESSES = 8
NUM_EPISODES = 100
DEVICE = torch.device("cpu")
BASE_STORAGE_DIR = root / "ssc_ppo_storage"

log = logging.getLogger()


def simulate_fn(
    envs: gym.vector.SyncVectorEnv,
    model: SSCPPONDNFBasedAgent,
    process_obs=Callable[[dict], Tensor],
    num_episodes=NUM_EPISODES,
) -> dict[str, Any]:
    logs: dict[str, Any] = {
        "num_frames_per_episode": [],
        "return_per_episode": [],
    }
    next_obs_dict, _ = envs.reset()
    next_obs = process_obs(next_obs_dict)
    next_obs_dict = {"input": next_obs}

    log_done_counter = 0
    log_episode_return = torch.zeros(NUM_PROCESSES, device=DEVICE)
    log_episode_num_frames = torch.zeros(NUM_PROCESSES, device=DEVICE)

    while log_done_counter < num_episodes:
        with torch.no_grad():
            actions = model.get_actions(
                preprocessed_obs=next_obs_dict, use_argmax=False
            )
        next_obs_dict, reward, terminations, truncations, _ = envs.step(
            actions[0]
        )
        next_obs = process_obs(next_obs_dict)
        next_obs_dict = {"input": next_obs}
        next_done = np.logical_or(terminations, truncations)

        log_episode_return += torch.tensor(
            reward, device=DEVICE, dtype=torch.float
        )
        log_episode_num_frames += torch.ones(NUM_PROCESSES, device=DEVICE)

        for i, done in enumerate(next_done):
            if done:
                log_done_counter += 1
                logs["return_per_episode"].append(log_episode_return[i].item())
                logs["num_frames_per_episode"].append(
                    log_episode_num_frames[i].item()
                )

        mask = 1 - torch.tensor(next_done, device=DEVICE, dtype=torch.float)
        log_episode_return *= mask
        log_episode_num_frames *= mask

    all_possible_wall_status = torch.Tensor(
        [
            [-1, -1],  # no wall on either side
            [1, -1],  # wall on the left
            [-1, 1],  # wall on the right
        ],
        device=DEVICE,
    ).float()

    with torch.no_grad():
        action_distribution = model.get_action_distribution(
            {"input": all_possible_wall_status}
        )
    logs["action_distribution"] = action_distribution.probs

    return logs


def post_training(
    model_dir: Path,
    model: SSCPPONDNFMutexTanhAgent,
) -> None:
    # Load from second prune after thresholding
    # Check for checkpoints first, this should exist
    assert (
        model_dir / "model_2nd_mr_pruned.pth"
    ).exists(), (
        "No 2nd pruned model found, please run the multirun_eval script first"
    )

    pruned_state = torch.load(
        model_dir / "model_2nd_mr_pruned.pth", map_location=DEVICE
    )
    model.load_state_dict(pruned_state)

    # Step 4: Raw enumeration of the disjunctions
    # -  This step needs to compute the bias of the disjunction layer
    conjunction_tensor = model.actor.conjunctions.weights.data.clone()
    disjunction_tensor = model.actor.disjunctions.weights.data.clone()

    disj_abs = disjunction_tensor.abs()
    disj_bias = model.actor.disjunctions._bias_calculation(disj_abs)

    # Step 5: Condensation via logical equivalence
    # Check if any conjunctions are equivalent
    equivalent_tuples = []
    for i in range(len(conjunction_tensor)):
        for j in range(i + 1, len(conjunction_tensor)):
            if torch.all(conjunction_tensor[i] == conjunction_tensor[j]):
                equivalent_tuples.append((i, j))

    # Step 6: Rule simplification based on experienced observations

    # Step 7: Interpretation of conjunction based on experienced observations

    # Step 8: ProbLog rules with annotated disjunction based on experienced observations


def post_train_eval(eval_cfg: DictConfig):
    experiment_name = eval_cfg["experiment_name"]
    use_ndnf = "ndnf" in experiment_name
    assert use_ndnf

    use_state_no_as_obs = "sn" in experiment_name
    assert not use_state_no_as_obs, "Has to use wall status as observation"

    envs = gym.vector.SyncVectorEnv(
        [make_env(eval_cfg, i, i, False) for i in range(NUM_PROCESSES)]
    )
    single_env = construct_single_environment(eval_cfg)
    process_obs = lambda obs: ss_corridor_preprocess_obs(
        use_state_no_as_obs=use_state_no_as_obs,
        use_ndnf=use_ndnf,
        corridor_length=single_env.corridor_length,
        obs=obs,
        device=DEVICE,
    )

    num_inputs = single_env.corridor_length if use_state_no_as_obs else 2

    for s in eval_cfg["multirun_seeds"]:
        # Load agent
        model_dir = BASE_STORAGE_DIR / f"{experiment_name}_{s}"
        model = construct_model(
            num_inputs=num_inputs,
            num_latent=eval_cfg["model_latent_size"],
            action_size=int(single_env.action_space.n),
            use_ndnf=use_ndnf,
            use_eo="eo" in experiment_name,
            use_mt="mt" in experiment_name,
        )
        assert isinstance(model, SSCPPONDNFMutexTanhAgent)
        model.to(DEVICE)
        model_state = torch.load(model_dir / "model.pth", map_location=DEVICE)
        model.load_state_dict(model_state)
        model.eval()

        log.info(f"Experiment {model_dir.name} loaded!")
        post_training(
            model_dir=model_dir,
            model=model,
        )

        log.info("======================================\n")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run_eval(cfg: DictConfig) -> None:
    eval_cfg = cfg["eval"]

    # Set random seed
    torch.manual_seed(DEFAULT_GEN_SEED)
    np.random.seed(DEFAULT_GEN_SEED)
    random.seed(DEFAULT_GEN_SEED)

    torch.autograd.set_detect_anomaly(True)  # type: ignore

    use_discord_webhook = cfg["webhook"]["use_discord_webhook"]
    msg_body = None
    errored = False
    keyboard_interrupt = None

    try:
        post_train_eval(eval_cfg)
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
                experiment_name=f"{eval_cfg['experiment_name']} Multirun Eval",
                message_body=msg_body,
                errored=errored,
                keyboard_interrupt=keyboard_interrupt,
            )


if __name__ == "__main__":
    import multiprocessing as mp

    if mp.get_start_method() != "forkserver":
        mp.set_start_method("forkserver", force=True)

    run_eval()
