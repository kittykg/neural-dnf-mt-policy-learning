# This script evaluates PPO agents on SpecialStateCorridor envs
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


from common import synthesize
from eval.common import METRIC_TO_SYMBOL_MAP
from ss_corridor_ppo import (
    construct_model,
    construct_single_environment,
    make_env,
    ss_corridor_preprocess_obs,
    SSCPPOBaseAgent,
)
from utils import post_to_discord_webhook


BASE_STORAGE_DIR = root / "ssc_ppo_storage"
DEFAULT_GEN_SEED = 2
DEVICE = torch.device("cpu")
NUM_EPISODES = 1000000
NUM_PROCESSES = 8

log = logging.getLogger()


def simulate(
    envs: gym.vector.SyncVectorEnv,
    model: SSCPPOBaseAgent,
    process_obs: Callable[[dict], Tensor],
    num_episodes: int = NUM_EPISODES,
    use_argmax: bool = True,
) -> dict[str, Any]:
    logs = {"num_frames_per_episode": [], "return_per_episode": []}
    next_obs_dict, _ = envs.reset()
    next_obs = process_obs(next_obs_dict)
    next_obs_dict = {"input": next_obs}

    log_done_counter = 0
    log_episode_return = torch.zeros(NUM_PROCESSES, device=DEVICE)
    log_episode_num_frames = torch.zeros(NUM_PROCESSES, device=DEVICE)

    while log_done_counter < num_episodes:
        with torch.no_grad():
            actions = model.get_actions(
                preprocessed_obs=next_obs_dict, use_argmax=use_argmax
            )
            if isinstance(actions, tuple):
                # NDNF-based agents return a tuple, where the first element is
                # the action sampled/argmax-ed from the action probabilities
                actions = actions[0]

        next_obs_dict, reward, terminations, truncations, _ = envs.step(actions)
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

    return logs


def rl_performance_eval(
    model: SSCPPOBaseAgent,
    envs: gym.vector.SyncVectorEnv,
    process_obs: Callable[[dict], Tensor],
    use_argmax: bool = True,
) -> dict[str, Any]:
    logs = simulate(
        envs=envs,
        model=model,
        process_obs=process_obs,
        num_episodes=NUM_EPISODES,
        use_argmax=use_argmax,
    )

    num_frames = sum(logs["num_frames_per_episode"])
    return_per_episode = synthesize(
        logs["return_per_episode"], compute_ste=True
    )
    num_frames_per_episode = synthesize(
        logs["num_frames_per_episode"], compute_ste=True
    )

    log_str = f"F {num_frames} | "
    log_str += "R: "
    for k, v in return_per_episode.items():
        log_str += f"{k} {v:.2f} "
    log_str += "| F: "
    for k, v in num_frames_per_episode.items():
        log_str += f"{k} {v:.1f} "
    log.info(log_str)

    logs["avg_return_per_episode"] = return_per_episode["mean"]

    return logs


def post_train_eval(eval_cfg: DictConfig) -> dict[str, float]:
    experiment_name = eval_cfg["experiment_name"]
    use_state_no_as_obs = "sn" in experiment_name
    use_ndnf = "ndnf" in experiment_name
    use_eo = "eo" in experiment_name
    use_mt = "mt" in experiment_name

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

    return_per_episode_list = []
    for s in eval_cfg["multirun_seeds"]:
        # Load agent
        model_dir = BASE_STORAGE_DIR / f"{experiment_name}_{s}"
        model = construct_model(
            num_inputs=num_inputs,
            num_latent=eval_cfg["model_latent_size"],
            action_size=int(single_env.action_space.n),
            use_ndnf=use_ndnf,
            use_eo=use_eo,
            use_mt=use_mt,
        )
        model.to(DEVICE)
        model_state = torch.load(model_dir / "model.pth", map_location=DEVICE)
        model.load_state_dict(model_state)
        model.eval()

        log.info(f"Experiment {model_dir.name} loaded!")
        ret_log = rl_performance_eval(
            model=model,
            envs=envs,
            process_obs=process_obs,
            use_argmax=eval_cfg.get("use_argmax", True),
        )
        return_per_episode_list.append(ret_log["avg_return_per_episode"])
        log.info("======================================\n")

    return_per_episode = synthesize(return_per_episode_list, compute_ste=True)
    log_str = f"{experiment_name} multi-run aggregated:  R - "
    for k, v in return_per_episode.items():
        log_str += f"{METRIC_TO_SYMBOL_MAP[k]} {v:.3f} "
    log.info(log_str)

    with open("aggregated_log.json", "w") as f:
        json.dump(
            {k: float(v) for k, v in return_per_episode.items()}, f, indent=4
        )

    return return_per_episode


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
        logs = post_train_eval(eval_cfg)
        if use_discord_webhook:
            msg_body = "Success!"
            for k, v in logs.items():
                msg_body += f"\nEpisodic return {k}: {v:.3f}"
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
    torch.set_warn_always(False)

    import multiprocessing as mp

    if mp.get_start_method() != "forkserver":
        mp.set_start_method("forkserver", force=True)

    run_eval()
