# This script evaluates the performance of the trained models on the Blackjack
# environment and compare the actor to a target policy from Sutton and Barto's
# book.
from collections import OrderedDict
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

from blackjack_common import *
from eval.blackjack_ppo_rl_eval_common import *
from utils import post_to_discord_webhook

DEFAULT_GEN_SEED = 2
BASE_STORAGE_DIR = root / "blackjack_ppo_storage"
DEVICE = torch.device("cpu")
log = logging.getLogger()


def single_model_eval(
    model: BlackjackBaseAgent,
    full_experiment_name: str,
    target_policy_csv_path: Path,
    device: torch.device,
    use_argmax: bool = True,
) -> dict[str, Any]:
    use_ndnf = "ndnf" in full_experiment_name
    if isinstance(model, BlackjackNDNFEOAgent):
        eval_model = model.to_ndnf_agent()
    else:
        eval_model = model
    eval_model.eval()

    env_eval_log = eval_on_environments(
        eval_model, device, use_argmax=use_argmax
    )

    if use_ndnf:
        assert isinstance(eval_model, BlackjackNDNFBasedAgent)
        cmp_q_eval_log = ndnf_based_agent_cmp_target_csv(
            target_policy_csv_path, eval_model, device
        )
    else:
        assert isinstance(eval_model, BlackjackMLPAgent)
        cmp_q_eval_log = mlp_agent_cmp_target_csv(
            target_policy_csv_path, eval_model, device
        )

    final_log = {**env_eval_log, **cmp_q_eval_log}
    return final_log


def result_analysis(
    single_eval_results: list[OrderedDict[str, float]],
) -> dict[str, float]:
    aggregated_log: dict[str, float] = dict()

    num_models = len(single_eval_results)
    all_runs_return = np.array(
        [d["avg_return_per_episode"] for d in single_eval_results]
    )
    all_runs_policy_error_cmp_to_q = np.array(
        [d["policy_error_cmp_to_q"] for d in single_eval_results]
    )
    all_runs_action_diversity_score = np.array(
        [d["action_diversity_score"] for d in single_eval_results]
    )
    all_runs_action_entropy = np.array(
        [d["action_entropy"] for d in single_eval_results]
    )
    all_runs_win_rate = np.array([d["win_rate"] for d in single_eval_results])

    def compute_mean_std_ste(arr: np.ndarray) -> tuple[float, float, float]:
        avg = float(np.mean(arr))
        std = float(np.std(arr))
        ste = float(std / np.sqrt(num_models))
        return avg, std, ste

    # Avg. return of all models
    avg_return, std_return, ste_return = compute_mean_std_ste(all_runs_return)
    log.info(f"Avg. return per episode of all runs: {avg_return:.3f}")
    log.info(f"Std. return per episode of all runs: {std_return:.3f}")
    log.info(f"Ste. return per episode of all runs: {ste_return:.3f}")
    log.info("=====================================")

    aggregated_log["avg_return_per_episode"] = avg_return
    aggregated_log["std_return_per_episode"] = std_return
    aggregated_log["ste_return_per_episode"] = ste_return

    # Policy error compared to Q of all models
    (
        avg_policy_error_cmp_to_q,
        std_policy_error_cmp_to_q,
        ste_policy_error_cmp_to_q,
    ) = compute_mean_std_ste(all_runs_policy_error_cmp_to_q)
    log.info(
        f"Avg. policy error compared to target Q of all runs: {avg_policy_error_cmp_to_q:.3f}"
    )
    log.info(
        f"Std. policy error compared to target Q of all runs: {std_policy_error_cmp_to_q:.3f}"
    )
    log.info(
        f"Ste. policy error compared to target Q of all runs: {ste_policy_error_cmp_to_q:.3f}"
    )
    log.info("=====================================")

    aggregated_log["avg_policy_error_cmp_to_q"] = avg_policy_error_cmp_to_q
    aggregated_log["std_policy_error_cmp_to_q"] = std_policy_error_cmp_to_q
    aggregated_log["ste_policy_error_cmp_to_q"] = ste_policy_error_cmp_to_q

    # Action diversity score of all models
    (
        avg_action_diversity_score,
        std_action_diversity_score,
        ste_action_diversity_score,
    ) = compute_mean_std_ste(all_runs_action_diversity_score)
    log.info(
        f"Avg. action diversity score of all runs: {avg_action_diversity_score:.3f}"
    )
    log.info(
        f"Std. action diversity score of all runs: {std_action_diversity_score:.3f}"
    )
    log.info(
        f"Ste. action diversity score of all runs: {ste_action_diversity_score:.3f}"
    )
    log.info("=====================================")

    aggregated_log["avg_action_diversity_score"] = avg_action_diversity_score
    aggregated_log["std_action_diversity_score"] = std_action_diversity_score
    aggregated_log["ste_action_diversity_score"] = ste_action_diversity_score

    # Action entropy of all models
    avg_action_entropy, std_action_entropy, ste_action_entropy = (
        compute_mean_std_ste(all_runs_action_entropy)
    )
    log.info(f"Avg. action entropy of all runs: {avg_action_entropy:.3f}")
    log.info(f"Std. action entropy of all runs: {std_action_entropy:.3f}")
    log.info(f"Ste. action entropy of all runs: {ste_action_entropy:.3f}")
    log.info("=====================================")

    aggregated_log["avg_action_entropy"] = avg_action_entropy
    aggregated_log["std_action_entropy"] = std_action_entropy
    aggregated_log["ste_action_entropy"] = ste_action_entropy

    # Win rate of all models
    avg_win_rate, std_win_rate, ste_win_rate = compute_mean_std_ste(
        all_runs_win_rate
    )
    log.info(f"Avg. win rate of all runs: {avg_win_rate:.3f}")
    log.info(f"Std. win rate of all runs: {std_win_rate:.3f}")
    log.info(f"Ste. win rate of all runs: {ste_win_rate:.3f}")

    aggregated_log["avg_win_rate"] = avg_win_rate
    aggregated_log["std_win_rate"] = std_win_rate
    aggregated_log["ste_win_rate"] = ste_win_rate
    log.info("=====================================")

    if "mutual_exclusivity" in single_eval_results[0]:
        all_runs_me = np.array(
            [d["mutual_exclusivity"] for d in single_eval_results]
        )
        all_runs_me_violations_count = np.array(
            [
                d["mutual_exclusivity_violations_count"]
                for d in single_eval_results
            ]
        )
        all_runs_ma = np.array(
            [d["missing_actions"] for d in single_eval_results]
        )
        all_runs_ma_count = np.array(
            [d.get("missing_actions_count", 0) for d in single_eval_results]
        )

        # Mutual exclusivity of all models
        avg_mutual_exclusivity = float(np.mean(all_runs_me))
        log.info(
            f"Avg. mutual exclusivity of all runs: {avg_mutual_exclusivity:.3f}"
        )
        aggregated_log["avg_mutual_exclusivity"] = avg_mutual_exclusivity

        # Mutual exclusivity violations count of all models
        avg_me_violations_count, _, _ = compute_mean_std_ste(
            all_runs_me_violations_count
        )
        log.info(
            f"Avg. mutual exclusivity violations count of all runs: {avg_me_violations_count:.3f}"
        )
        log.info("=====================================")

        aggregated_log["avg_mutual_exclusivity_violations_count"] = (
            avg_me_violations_count
        )

        # Missing actions of all models
        avg_missing_actions = float(np.mean(all_runs_ma))
        log.info(f"Avg. missing actions of all runs: {avg_missing_actions:.3f}")
        aggregated_log["avg_missing_actions"] = avg_missing_actions

        # Missing actions count of all models
        avg_missing_actions_count, _, _ = compute_mean_std_ste(
            all_runs_ma_count
        )
        log.info(
            f"Avg. missing actions count of all runs: {avg_missing_actions_count:.3f}"
        )
        log.info("=====================================")

        aggregated_log["avg_missing_actions_count"] = avg_missing_actions_count

    with open("aggregated_log.json", "w") as f:
        json.dump(aggregated_log, f, indent=4)

    return aggregated_log


def multirun_rl_performance_eval(eval_cfg: DictConfig) -> dict[str, Any]:
    experiment_name = f"{eval_cfg['experiment_name']}"
    use_ndnf = "ndnf" in experiment_name
    use_argmax = eval_cfg.get("use_argmax", True)

    target_policy_csv_path = Path(eval_cfg["target_policy_csv_path"])
    if not target_policy_csv_path.exists():
        raise FileNotFoundError(
            f"The target policy csv file {target_policy_csv_path} does not exist!"
        )

    single_eval_results = []

    for s in eval_cfg["multirun_seeds"]:
        # Load agent
        full_experiment_name = f"{experiment_name}_{s}"
        model_dir = BASE_STORAGE_DIR / full_experiment_name
        model = construct_model(
            num_latent=eval_cfg["model_latent_size"],
            use_ndnf=use_ndnf,
            use_decode_obs=True,
            use_eo=False,
            use_mt=True,
            share_layer_with_critic=eval_cfg["share_layer_with_critic"],
        )
        model.to(DEVICE)
        model_state = torch.load(
            model_dir / "model.pth", map_location=DEVICE, weights_only=True
        )
        model.load_state_dict(model_state)
        model.eval()

        log.info(f"Experiment {model_dir.name} loaded!")

        eval_log = single_model_eval(
            model,
            full_experiment_name,
            target_policy_csv_path,
            DEVICE,
            use_argmax=use_argmax,
        )
        single_eval_results.append(eval_log)

    log.info("Evaluation finished!")
    log.info(
        f"Results of {eval_cfg['experiment_name']} (argmax: {use_argmax}):"
    )
    aggregated_log = result_analysis(
        single_eval_results,
    )
    return aggregated_log


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run_eval(cfg: DictConfig) -> None:
    eval_cfg = cfg["eval"]

    # Set random seed
    torch.manual_seed(DEFAULT_GEN_SEED)
    np.random.seed(DEFAULT_GEN_SEED)
    random.seed(DEFAULT_GEN_SEED)

    torch.autograd.set_detect_anomaly(True)

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
