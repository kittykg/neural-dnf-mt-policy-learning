# This script evaluates the NDNF MT's ASP rules on the DoorCorridor environment
import json
import logging
from pathlib import Path
import random
import sys
import traceback


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


from corridor_grid.envs.base_ss_corridor import BaseSpecialStateCorridorEnv

from common import synthesize
from eval.asp_inference_common import (
    ASPRuleEvaluationFailureCode,
    evaluate_rule_on_env,
)
from eval.door_corridor_ppo_ndnf_mt_multirun_eval import ASP_RULES_FILE_NAME
from ss_corridor_ppo import construct_single_environment
from utils import post_to_discord_webhook


BASE_STORAGE_DIR = root / "ssc_ppo_storage"
DEFAULT_GEN_SEED = 2
DEVICE = torch.device("cpu")
NUM_EVAL_EPISODES = 10


log = logging.getLogger()


def load_asp_rules(eval_cfg: DictConfig, seed: int) -> None | list[str]:
    experiment_name = eval_cfg["experiment_name"]
    model_dir = BASE_STORAGE_DIR / f"{experiment_name}_{seed}"
    if not (model_dir / ASP_RULES_FILE_NAME).exists():
        return None

    with open(model_dir / ASP_RULES_FILE_NAME, "r") as f:
        asp_rules = f.readlines()
    # remove the newline character
    asp_rules = [rule.strip() for rule in asp_rules]

    return asp_rules


def rule_eval(
    env: BaseSpecialStateCorridorEnv, asp_rules: list[str]
) -> ASPRuleEvaluationFailureCode | list[float]:
    def context_generation(obs: dict[str, np.ndarray]) -> list[str]:
        return [f"a_{obs['agent_location']}."]

    return evaluate_rule_on_env(
        env=env,
        context_encoding_generation_fn=context_generation,
        num_actions=2,
        rules=asp_rules,
        eval_num_episodes=NUM_EVAL_EPISODES,
        do_logging=False,
    )


def post_train_eval(eval_cfg: DictConfig) -> None:
    experiment_name = f"{eval_cfg['experiment_name']}"
    use_ndnf = "ndnf" in experiment_name

    assert use_ndnf

    env = construct_single_environment(eval_cfg)

    skipped_seeds = []
    errored_seeds = []
    successful_seeds = []
    all_returns = []

    for s in eval_cfg["multirun_seeds"]:
        # Load agent
        ret = load_asp_rules(eval_cfg, s)
        if ret is None:
            log.info(
                f"ASP rules file not found for {experiment_name}_{s}, "
                "skipping..."
            )
            skipped_seeds.append(s)
            continue

        asp_rules = ret

        log.info(f"Experiment {experiment_name}_{s} loaded!")
        ret = rule_eval(env, asp_rules)
        if isinstance(ret, ASPRuleEvaluationFailureCode):
            errored_seeds.append(s)
        else:
            successful_seeds.append(s)
            synth_log = synthesize(ret, compute_ste=True)
            for k, v in synth_log.items():
                log.info(f"  {k}: {v:.2f}")
            all_returns.append(ret)

        log.info("======================================")

    log.info(f"Skipped seeds: {skipped_seeds}")
    log.info(f"Errored seeds: {errored_seeds}")
    log.info(f"Successful seeds: {successful_seeds}")
    log.info("======================================")

    log.info("Successful seeds aggregated log:")
    flatten_returns = np.array([i for l in all_returns for i in l])
    aggregated_log = synthesize(flatten_returns, compute_ste=True)
    for k, v in aggregated_log.items():
        log.info(f"  {k}: {v:.3f}")

    json_dict = {}
    json_dict["skipped_seeds"] = skipped_seeds
    json_dict["errored_seeds"] = errored_seeds
    json_dict["successful_seeds"] = successful_seeds
    json_dict["aggregated_log"] = {
        k: float(v) for k, v in aggregated_log.items()
    }
    with open("asp_eval_aggregated_log.json", "w") as f:
        json.dump(json_dict, f, indent=4)


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
    torch.set_warn_always(False)

    import multiprocessing as mp

    if mp.get_start_method() != "forkserver":
        mp.set_start_method("forkserver", force=True)

    run_eval()
