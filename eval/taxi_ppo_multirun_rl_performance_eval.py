# This script evaluates the performance of the trained models on the Taxi
# environment.
import json
import logging
from pathlib import Path
import random
import sys
import traceback
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig
import torch

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
from eval.taxi_ppo_rl_eval_common import eval_model_on_environment
from taxi_common import (
    TaxiEnvPPOBaseAgent,
    TaxiEnvPPONDNFEOAgent,
    construct_model,
)
from utils import post_to_discord_webhook

BASE_STORAGE_DIR = root / "taxi_ppo_storage"
DEFAULT_GEN_SEED = 2
DEVICE = torch.device("cpu")
log = logging.getLogger()


def single_model_eval(
    model: TaxiEnvPPOBaseAgent,
    device: torch.device,
    use_argmax: bool = True,
    eval_num_runs: int = 10000,
) -> dict[str, Any]:

    if isinstance(model, TaxiEnvPPONDNFEOAgent):
        eval_model = model.to_ndnf_agent()
    else:
        eval_model = model
    eval_model.eval()

    logs = eval_model_on_environment(
        eval_model, device, use_argmax, eval_num_runs
    )

    num_frames = sum(logs["num_frames_per_episode"])
    return_per_episode = synthesize(
        logs["return_per_episode"], compute_ste=True
    )
    num_frames_per_episode = synthesize(
        logs["num_frames_per_episode"], compute_ste=True
    )

    log_str = f"{'Argmax' if use_argmax else 'Non-argmax'}: "
    log_str += f"F {num_frames} | "
    log_str += "R: "
    for k, v in return_per_episode.items():
        log_str += f"{k} {v:.2f} "
    log_str += "| F: "
    for k, v in num_frames_per_episode.items():
        log_str += f"{k} {v:.1f} "
    log.info(log_str)

    logs["avg_return_per_episode"] = return_per_episode["mean"]

    return logs


def multirun_rl_performance_eval(eval_cfg: DictConfig) -> dict[str, Any]:
    experiment_name = f"{eval_cfg['experiment_name']}"
    use_ndnf = "ndnf" in experiment_name

    argmax_return_per_episode_list = []
    non_argmax_return_per_episode_list = []

    for s in eval_cfg["multirun_seeds"]:
        # Load agent
        full_experiment_name = f"{experiment_name}_{s}"
        model_dir = BASE_STORAGE_DIR / full_experiment_name
        model = construct_model(
            actor_latent_size=eval_cfg["actor_latent_size"],
            use_ndnf=use_ndnf,
            use_decode_obs=eval_cfg["use_decode_obs"],
            use_eo="use_eo" in eval_cfg and eval_cfg["use_eo"],
            use_mt="use_mt" in eval_cfg and eval_cfg["use_mt"],
            share_layer_with_critic=eval_cfg.get(
                "share_layer_with_critic", False
            ),
            critic_latent_1=eval_cfg.get("critic_latent_1", 256),
            critic_latent_2=eval_cfg.get("critic_latent_2", 64),
            pretrained_critic=eval_cfg.get("pretrained_critic", None),
            mlp_actor_disable_bias=eval_cfg.get(
                "mlp_actor_disable_bias", False
            ),
        )
        model.to(DEVICE)
        model_state = torch.load(model_dir / "model.pth", map_location=DEVICE)
        model.load_state_dict(model_state)
        model.eval()

        log.info(f"Experiment {model_dir.name} loaded!")

        argmax_log = single_model_eval(model, DEVICE, True)
        non_argmax_log = single_model_eval(model, DEVICE, False)

        argmax_return_per_episode_list.append(
            argmax_log["avg_return_per_episode"]
        )
        non_argmax_return_per_episode_list.append(
            non_argmax_log["avg_return_per_episode"]
        )
        log.info("======================================\n")

    log.info("Evaluation finished!")
    log.info(f"Results of {eval_cfg['experiment_name']}:")

    avg_return_per_episode = synthesize(
        argmax_return_per_episode_list, compute_ste=True
    )
    avg_return_per_episode_non_argmax = synthesize(
        non_argmax_return_per_episode_list, compute_ste=True
    )

    final_log: dict[str, float] = {}

    log.info(f"{experiment_name} multi-run aggregated:")
    log_str = "Argmax R - "
    for k, v in avg_return_per_episode.items():
        log_str += f"{METRIC_TO_SYMBOL_MAP[k]} {v:.3f} "
        final_log[f"argmax_return_per_episode_{k}"] = v
    log.info(log_str)

    log_str = "Non-argmax R - "
    for k, v in avg_return_per_episode_non_argmax.items():
        log_str += f"{METRIC_TO_SYMBOL_MAP[k]} {v:.3f} "
        final_log[f"non_argmax_return_per_episode_{k}"] = v
    log.info(log_str)

    with open("rl_perf_aggregated_log.json", "w") as f:
        json.dump(final_log, f, indent=4)

    return final_log


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run_eval(cfg: DictConfig) -> None:
    eval_cfg = cfg["eval"]

    # Set random seed
    torch.manual_seed(DEFAULT_GEN_SEED)
    np.random.seed(DEFAULT_GEN_SEED)
    random.seed(DEFAULT_GEN_SEED)

    # torch.autograd.set_detect_anomaly(True)

    use_discord_webhook = cfg["webhook"]["use_discord_webhook"]
    msg_body = None
    errored = False
    keyboard_interrupt = None

    try:
        aggregated_log = multirun_rl_performance_eval(eval_cfg)
        if use_discord_webhook:
            msg_body = "Success!"
            for k, v in aggregated_log.items():
                msg_body += f"\n{k}: {v:.3f}"
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
                experiment_name=eval_cfg["experiment_name"],
                message_body=msg_body,
                errored=errored,
                keyboard_interrupt=keyboard_interrupt,
            )


if __name__ == "__main__":
    torch.set_warn_always(False)

    import multiprocessing as mp

    if mp.get_start_method() != "fork":
        mp.set_start_method("fork", force=True)

    run_eval()
