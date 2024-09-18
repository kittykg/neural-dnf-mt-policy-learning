# This script soft extracts (prune and threshold on conjunctions) the NDNF-MT
# actor trained on the Blackjack environment, based on the comparison result on
# a target Q-value table. This script is the pre-requisite for the ProbLog
# interpretation script.
from copy import deepcopy
import json
import logging
from pathlib import Path
import random
import sys
import traceback
from typing import Any

import numpy as np
import hydra
from omegaconf import DictConfig
import torch
import torch.nn.functional as F

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from neural_dnf.post_training import prune_neural_dnf

from blackjack_common import (
    construct_model,
    get_target_policy,
    BlackjackNDNFMutexTanhAgent,
)
from eval.blackjack_ppo_rl_eval_common import ndnf_based_agent_cmp_target_csv
from utils import post_to_discord_webhook


DEFAULT_GEN_SEED = 2
DEVICE = torch.device("cpu")
BASE_STORAGE_DIR = root / "blackjack_ppo_storage"

FIRST_PRUNE_MODEL_PTH_NAME = "model_soft_mr_pruned.pth"
THRESHOLD_MODEL_PTH_NAME = "soft_thresholded_model.pth"
THRESHOLD_JSON_NAME = "soft_threshold_val_candidates.json"
SECOND_PRUNE_MODEL_PTH_NAME = "model_soft_2nd_mr_pruned.pth"

log = logging.getLogger()


def post_training(
    model: BlackjackNDNFMutexTanhAgent,
    target_policy_csv_path: Path,
    model_dir: Path,
) -> dict[str, Any]:
    target_policy = get_target_policy(target_policy_csv_path)
    total_number_of_states = len(target_policy)

    # Helper functions
    def _simulate_with_print(model_name: str) -> dict[str, Any]:
        logs = ndnf_based_agent_cmp_target_csv(
            target_policy_csv_path, model, DEVICE
        )
        log.info(f"Model: {model_name}")
        log.info(
            f"No. ME violations: {logs.get('mutual_exclusivity_violations_count', 0)}"
        )
        log.info(f"No. missing actions: {logs.get('missing_actions_count', 0)}")
        log.info(f"Policy error compared to Q: {logs['policy_error_cmp_to_q']}")
        log.info(
            f"Normal states percentage: {len(logs['normal_indices']) / total_number_of_states}"
        )
        return logs

    # Stage 1: Evaluate the model post-training only on normal states
    _simulate_with_print("NDNF MT Soft")
    log.info("======================================")

    # Stage 2: Prune the model
    def prune_model() -> None:
        log.info("Pruning the model...")

        def pruning_cmp_fn(og_log, new_log):
            # Check if the action distribution is close
            # We don't use kl divergence just yet
            og_action_dist = og_log["action_distribution"]
            new_action_dist = new_log["action_distribution"]

            return np.allclose(og_action_dist, new_action_dist, atol=1e-3)

        sd_list = []
        prune_count = 0

        while True:
            log.info(f"Pruning iteration: {prune_count + 1}")
            prune_result_dict = prune_neural_dnf(
                model.actor,
                ndnf_based_agent_cmp_target_csv,
                {
                    "target_policy_csv_path": target_policy_csv_path,
                    "agent": model,
                    "device": DEVICE,
                },
                pruning_cmp_fn,
                options={
                    "skip_prune_disj_with_empty_conj": False,
                    "skip_last_prune_disj": True,
                },
            )

            important_keys = [
                "disj_prune_count_1",
                "unused_conjunctions_2",
                "conj_prune_count_3",
                "prune_disj_with_empty_conj_count_4",
            ]

            log.info(
                f"Pruned disjunction count: {prune_result_dict['disj_prune_count_1']}"
            )
            log.info(
                f"Removed unused conjunction count: {prune_result_dict['unused_conjunctions_2']}"
            )
            log.info(
                f"Pruned conjunction count: {prune_result_dict['conj_prune_count_3']}"
            )
            log.info(
                f"Pruned disj with empty conj: {prune_result_dict['prune_disj_with_empty_conj_count_4']}"
            )

            log.info("-------------")
            # If any of the important keys has the value not 0, then we should continue pruning
            if any([prune_result_dict[k] != 0 for k in important_keys]):
                sd_list.append(deepcopy(model.state_dict()))
                prune_count += 1
            else:
                break

    # Check for checkpoints
    # If the model is already pruned, then we load the pruned model
    # Otherwise, we prune the model and save the pruned model
    if (model_dir / FIRST_PRUNE_MODEL_PTH_NAME).exists():
        pruned_state = torch.load(
            model_dir / FIRST_PRUNE_MODEL_PTH_NAME, map_location=DEVICE
        )
        model.load_state_dict(pruned_state)
    else:
        prune_model()
        torch.save(model.state_dict(), model_dir / FIRST_PRUNE_MODEL_PTH_NAME)

    post_prune_logs = _simulate_with_print("NDNF MT Soft pruned")

    log.info("======================================")

    # 3. Thresholding
    og_conj_weight = model.actor.conjunctions.weights.data.clone()

    def threshold_model() -> dict[str, float]:
        log.info("Thresholding the model conjunction...")

        conj_min = torch.min(model.actor.conjunctions.weights.data)
        conj_max = torch.max(model.actor.conjunctions.weights.data)
        threshold_upper_bound = round(
            (torch.Tensor([conj_min, conj_max]).abs().max() + 0.01).item(),
            2,
        )
        log.info(f"Threshold upper bound: {threshold_upper_bound}")

        t_vals = torch.arange(0, threshold_upper_bound, 0.01)
        result_dicts = []

        for v in t_vals:
            model.actor.conjunctions.weights.data = (
                (torch.abs(og_conj_weight) > v)
                * torch.sign(og_conj_weight)
                * 6.0
            )
            r = ndnf_based_agent_cmp_target_csv(
                target_policy_csv_path, model, DEVICE
            )
            r["t_val"] = v.item()
            r["kl"] = F.kl_div(
                input=torch.log(torch.tensor(r["action_distribution"]) + 1e-8),
                target=torch.tensor(post_prune_logs["action_distribution"]),
                reduction="batchmean",
            ).item()
            result_dicts.append(r)

        log.info("Proceed to threshold based on KL...")
        second_sorted_result_dict = sorted(
            result_dicts,
            key=lambda d: (
                d["kl"],
                d["policy_error_cmp_to_q"],
            ),
        )
        best_candidate = second_sorted_result_dict[0]
        log.info(f"Best candidate: {best_candidate['t_val']}")
        log.info(f"KL: {best_candidate['kl']}")

        return {
            "t_val": best_candidate["t_val"],
            "kl": best_candidate["kl"],
        }

    # Check for checkpoints
    # If the thresholding process is done, then we load the threshold candidates
    # Otherwise, we threshold the model and save the threshold candidates
    def apply_threshold_with_candidate(t_val: float):
        log.info(f"Applying threshold: {t_val}")
        model.actor.conjunctions.weights.data = (
            (torch.abs(og_conj_weight) > t_val)
            * torch.sign(og_conj_weight)
            * 6.0
        )
        torch.save(model.state_dict(), model_dir / THRESHOLD_MODEL_PTH_NAME)

    if (model_dir / THRESHOLD_JSON_NAME).exists():
        with open(model_dir / THRESHOLD_JSON_NAME, "r") as f:
            threshold_json_dict = json.load(f)
        t_val = threshold_json_dict["threshold_val"]

        if (model_dir / THRESHOLD_MODEL_PTH_NAME).exists():
            thresholded_state = torch.load(
                model_dir / THRESHOLD_MODEL_PTH_NAME, map_location=DEVICE
            )
            model.load_state_dict(thresholded_state)
        else:
            apply_threshold_with_candidate(t_val)
    else:
        ret_dict = threshold_model()
        t_val = ret_dict["t_val"]

        with open(model_dir / THRESHOLD_JSON_NAME, "w") as f:
            threshold_json_dict = {}
            threshold_json_dict["threshold_val"] = t_val
            json.dump(threshold_json_dict, f)

        apply_threshold_with_candidate(t_val)

    _simulate_with_print("NDNF MT Soft (thresholded)")
    log.info("======================================")

    # Stage 4. Second prune after thresholding
    # Again, check for checkpoints first
    # If the model is already pruned, then we load the pruned model
    # Otherwise, we prune the model and save the pruned model
    if (model_dir / SECOND_PRUNE_MODEL_PTH_NAME).exists():
        pruned_state = torch.load(
            model_dir / SECOND_PRUNE_MODEL_PTH_NAME, map_location=DEVICE
        )
        model.load_state_dict(pruned_state)
    else:
        prune_model()
        torch.save(model.state_dict(), model_dir / SECOND_PRUNE_MODEL_PTH_NAME)

    second_prune_logs = _simulate_with_print(f"NDNF MT Soft 2nd prune")
    kl = F.kl_div(
        input=torch.log(
            torch.tensor(second_prune_logs["action_distribution"]) + 1e-8
        ),
        target=torch.tensor(post_prune_logs["action_distribution"]),
        reduction="batchmean",
    ).item()
    log.info(f"KL divergence cmp to after 1st prune: {kl}")

    return second_prune_logs


def post_train_eval(eval_cfg: DictConfig) -> dict[str, Any]:
    experiment_name = f"{eval_cfg['experiment_name']}"
    use_ndnf = "ndnf" in experiment_name

    assert use_ndnf and not eval_cfg["use_eo"] and eval_cfg["use_mt"]

    target_policy_csv_path = Path(eval_cfg["target_policy_csv_path"])
    if not target_policy_csv_path.exists():
        raise FileNotFoundError(
            f"The target policy csv file {target_policy_csv_path} does not exist!"
        )

    policy_error_cmp_to_q_list: list[float] = []

    for s in eval_cfg["multirun_seeds"]:
        # Load agent
        model_dir = BASE_STORAGE_DIR / f"{experiment_name}_{s}"
        model: BlackjackNDNFMutexTanhAgent = construct_model(
            num_latent=eval_cfg["model_latent_size"],
            use_ndnf=use_ndnf,
            use_decode_obs=True,
            use_eo=False,
            use_mt=True,
            share_layer_with_critic=eval_cfg["share_layer_with_critic"],
        )  # type: ignore
        model.to(DEVICE)
        model_state = torch.load(model_dir / "model.pth", map_location=DEVICE)
        model.load_state_dict(model_state)
        model.eval()

        log.info(f"Experiment {model_dir.name} loaded!")
        ret = post_training(model, target_policy_csv_path, model_dir)
        policy_error_cmp_to_q_list.append(ret["policy_error_cmp_to_q"])
        log.info("======================================\n")

    log.info(
        f"Average policy error compared to Q: {np.array(policy_error_cmp_to_q_list).mean()}"
    )

    return {
        "avg_policy_error_cmp_to_q": np.array(
            policy_error_cmp_to_q_list
        ).mean(),
    }


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
        ret_dict = post_train_eval(eval_cfg)
        if use_discord_webhook:
            msg_body = f"Success!\n"
            msg_body += f"Average policy error compared to Q: {ret_dict['avg_policy_error_cmp_to_q']}"
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
