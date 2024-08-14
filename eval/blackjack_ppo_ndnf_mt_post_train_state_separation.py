# This script soft-extract ASP rules from the NDNF MT model on the Blackjack
# environment, based on the comparison result on a target Q-value table.
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

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from neural_dnf.post_training import (
    prune_neural_dnf,
    apply_threshold,
    extract_asp_rules,
    get_thresholding_upper_bound,
)

from blackjack_common import (
    construct_model,
    construct_single_environment,
    get_target_policy,
    BlackjackNDNFMutexTanhAgent,
)
from eval.common import ToyTextSoftExtractionReturnCode
from eval.blackjack_ppo_rl_eval_common import ndnf_based_agent_cmp_target_csv
from utils import post_to_discord_webhook


DEFAULT_GEN_SEED = 2
BLACKJACK_SINGLE_ENV_NUM_EPISODES = 500
DEVICE = torch.device("cpu")
BASE_STORAGE_DIR = root / "blackjack_ppo_storage"

log = logging.getLogger()
single_env = construct_single_environment()


def post_training(
    model: BlackjackNDNFMutexTanhAgent,
    target_policy_csv_path: Path,
    eval_cfg: DictConfig,
    model_dir: Path,
) -> ToyTextSoftExtractionReturnCode | dict[str, Any]:
    soft_traction_return_codes = []
    target_policy = get_target_policy(target_policy_csv_path)
    total_number_of_states = len(target_policy)

    # Stage 0: Separate states that violates mutual exclusivity and missing
    # actions
    initial_log = ndnf_based_agent_cmp_target_csv(
        target_policy_csv_path, model, DEVICE
    )
    normal_indices = initial_log["normal_indices"]

    if initial_log["mutual_exclusivity"] and not initial_log["missing_actions"]:
        soft_traction_return_codes.append(
            ToyTextSoftExtractionReturnCode.AFTER_TRAIN_NO_ABNORMAL_STATES
        )

    log.info(f"Initial NDNF-MT agent eval:")
    log.info(f"Mutual exclusivity: {initial_log['mutual_exclusivity']}")
    log.info(f"Missing actions: {initial_log['missing_actions']}")
    log.info(
        f"Policy error compared to Q: {initial_log['policy_error_cmp_to_q']}"
    )
    log.info(f"Normal states indices: {normal_indices}")
    log.info(
        "The following processes are all done on these normal state indices. "
        "For the other states, we will process in a different way."
    )
    log.info("======================================")

    # Helper functions
    def _simulate_with_print(model_name: str) -> dict[str, Any]:
        logs = ndnf_based_agent_cmp_target_csv(
            target_policy_csv_path, model, DEVICE, normal_indices
        )
        log.info(f"Model: {model_name}")
        log.info(f"Mutual exclusivity: {logs['mutual_exclusivity']}")
        log.info(f"Missing actions: {logs['missing_actions']}")
        log.info(f"Policy error compared to Q: {logs['policy_error_cmp_to_q']}")
        return logs

    def comparison_fn(
        og_log,
        new_log,
        t_a: float = 1e-3,
        t_b: float = 1e-3,
        criterion: list[str] = [],
    ) -> bool:
        # Mandatory checks
        if not new_log["mutual_exclusivity"]:
            return False
        if new_log["missing_actions"]:
            return False

        criteria_a = (
            og_log["policy_error_cmp_to_q"] - new_log["policy_error_cmp_to_q"]
            >= -t_a
        )
        criteria_b = (
            np.count_nonzero(og_log["actions"] != new_log["actions"])
            / len(og_log["actions"])
            <= t_b
        )

        if criterion == ["a"]:
            return criteria_a
        if criterion == ["b"]:
            return criteria_b
        if criterion == ["a", "b"]:
            return criteria_a and criteria_b

        return True

    # Stage 1: Evaluate the model post-training only on normal states
    og_ndnf_mt_logs = _simulate_with_print("NDNF MT (normal states only)")
    log.info("======================================")

    # Stage 2: Prune the model
    def prune_model() -> ToyTextSoftExtractionReturnCode | None:
        log.info("Pruning the model...")

        pruning_cmp_option = eval_cfg.get("pruning_cmp_option", ["a", "b"])
        assert pruning_cmp_option in [[], ["a"], ["b"], ["a", "b"]]
        pruning_cmp_t_a = eval_cfg.get("pruning_cmp_t_a", 1e-3)
        pruning_cmp_t_b = eval_cfg.get("pruning_cmp_t_b", 1e-3)

        pruning_cmp_fn = lambda og_log, new_log: comparison_fn(
            og_log,
            new_log,
            t_a=pruning_cmp_t_a,
            t_b=pruning_cmp_t_b,
            criterion=pruning_cmp_option,
        )

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
                    "normal_indices": normal_indices,
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

            post_prune_ndnf_mt_log = _simulate_with_print(
                f"NDNF MT - (Prune iteration: {prune_count + 1})"
            )

            if post_prune_ndnf_mt_log["missing_actions"]:
                log.info("Post prune NDNF MT has missing actions!")
                return ToyTextSoftExtractionReturnCode.FAIL_AT_PRUNE_MISS_ACTION
            if not post_prune_ndnf_mt_log["mutual_exclusivity"]:
                log.info("Post prune NDNF MT is not mutually exclusive!")
                return ToyTextSoftExtractionReturnCode.FAIL_AT_PRUNE_NOT_ME

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
    if (model_dir / "model_mr_pruned.pth").exists():
        pruned_state = torch.load(
            model_dir / "model_mr_pruned.pth", map_location=DEVICE
        )
        model.load_state_dict(pruned_state)
    else:
        ret = prune_model()
        if ret is not None and isinstance(ret, ToyTextSoftExtractionReturnCode):
            return ret
        torch.save(model.state_dict(), model_dir / "model_mr_pruned.pth")

    _simulate_with_print("NDNF MT pruned")
    soft_traction_return_codes.append(
        ToyTextSoftExtractionReturnCode.AFTER_PRUNE_NO_ABNORMAL_STATES
    )
    log.info("======================================")

    # 3. Thresholding
    og_conj_weight = model.actor.conjunctions.weights.data.clone()
    og_disj_weight = model.actor.disjunctions.weights.data.clone()

    # Convert the agent to plain NDNF agent
    model = model.to_ndnf_agent()  # type: ignore
    model.eval()

    def threshold_model() -> (
        tuple[ToyTextSoftExtractionReturnCode, dict[str, list]]
    ):
        log.info("Thresholding the model...")

        threshold_upper_bound = get_thresholding_upper_bound(model.actor)
        log.info(f"Threshold upper bound: {threshold_upper_bound}")

        t_vals = torch.arange(0, threshold_upper_bound, 0.01)
        result_dicts = []

        for v in t_vals:
            apply_threshold(model.actor, og_conj_weight, og_disj_weight, v)
            r = ndnf_based_agent_cmp_target_csv(
                target_policy_csv_path, model, DEVICE, normal_indices
            )
            r["t_val"] = v.item()
            result_dicts.append(r)

        sorted_result_dict = sorted(
            result_dicts, key=lambda d: d["policy_error_cmp_to_q"]
        )
        t_vals_candidates = []

        thresholding_cmp_option = eval_cfg.get("thresholding_cmp_option", [])
        assert thresholding_cmp_option in [[], ["a"], ["b"], ["a", "b"]]
        thresholding_cmp_t_a = eval_cfg.get("thresholding_cmp_t_a", 1e-3)
        thresholding_cmp_t_b = eval_cfg.get("thresholding_cmp_t_b", 1e-3)

        # Check for perfect thresholding
        for d in sorted_result_dict:
            if not comparison_fn(
                og_ndnf_mt_logs,
                d,
                t_a=thresholding_cmp_t_a,
                t_b=thresholding_cmp_t_b,
                criterion=thresholding_cmp_option,
            ):
                continue
            t_vals_candidates.append(d["t_val"])

        if len(t_vals_candidates) != 0:
            log.info(
                f"t_vals_candidates: {[round(v, 2) for v in t_vals_candidates]}"
            )
            return (
                ToyTextSoftExtractionReturnCode.THRESHOLD_HAS_PERFECT_CANDIDATE,
                {
                    "t_val_candidates": t_vals_candidates,
                    "normal_indices": normal_indices,
                },
            )

        log.info("No perfect thresholding candidate found!")

        # Check for imperfect thresholding
        log.info("Proceed to soft threshold...")
        for d in sorted_result_dict:
            me_violation_count = d.get("mutual_exclusivity_violations_count", 0)
            ma_count = d.get("missing_actions_count", 0)
            combined_abnormal_count = me_violation_count + ma_count
            d["combined_abnormal_count"] = combined_abnormal_count

        second_sorted_result_dict = sorted(
            result_dicts,
            key=lambda d: (
                d["combined_abnormal_count"],
                d["policy_error_cmp_to_q"],
            ),
        )
        best_candidate = second_sorted_result_dict[0]
        log.info(f"Best candidate: {best_candidate['t_val']}")
        log.info(
            f"Combined abnormal count: {best_candidate['combined_abnormal_count']}"
        )
        final_normal_indices = np.array(normal_indices)[
            best_candidate["normal_indices"]
        ]

        return ToyTextSoftExtractionReturnCode.THRESHOLD_IMPERFECT_CANDIDATE, {
            "t_val_candidates": [best_candidate["t_val"]],
            "normal_indices": final_normal_indices.tolist(),
        }

    # Check for checkpoints
    # If the thresholding process is done, then we load the threshold candidates
    # Otherwise, we threshold the model and save the threshold candidates
    def apply_threshold_with_candidate_list(t_vals_candidates: list[float]):
        log.info(f"Applying threshold: {t_vals_candidates[0]}")
        apply_threshold(
            model.actor,
            og_conj_weight,
            og_disj_weight,
            t_vals_candidates[0],
        )
        torch.save(model.state_dict(), model_dir / "thresholded_model.pth")

    if (model_dir / "threshold_val_candidates.json").exists():
        with open(model_dir / "threshold_val_candidates.json", "r") as f:
            threshold_json_dict = json.load(f)

        ret_code = threshold_json_dict["threshold_ret_code"]
        t_vals_candidates = threshold_json_dict["threshold_vals"]
        normal_indices = threshold_json_dict["normal_indices"]

        if (model_dir / "thresholded_model.pth").exists():
            thresholded_state = torch.load(
                model_dir / "thresholded_model.pth", map_location=DEVICE
            )
            model.load_state_dict(thresholded_state)
        else:
            apply_threshold_with_candidate_list(t_vals_candidates)
    else:
        ret_code, ret_dict = threshold_model()
        t_vals_candidates = ret_dict["t_val_candidates"]
        normal_indices = ret_dict["normal_indices"]

        with open(model_dir / "threshold_val_candidates.json", "w") as f:
            threshold_json_dict = {}
            threshold_json_dict["threshold_ret_code"] = ret_code.name
            threshold_json_dict["threshold_vals"] = t_vals_candidates
            threshold_json_dict["normal_indices"] = normal_indices
            json.dump(threshold_json_dict, f)

        apply_threshold_with_candidate_list(t_vals_candidates)

    threshold_log = _simulate_with_print("NDNF MT (thresholded)")
    soft_traction_return_codes.append(ret_code)
    log.info("======================================")

    # 4. Rule extraction
    if (model_dir / "asp_rules.lp").exists():
        with open(model_dir / "asp_rules.lp", "r") as f:
            rules = f.readlines()
    else:
        rules: list[str] = extract_asp_rules(model.actor.state_dict())  # type: ignore
        with open(model_dir / "asp_rules.lp", "w") as f:
            f.write("\n".join(rules))
    for r in rules:
        log.info(r.strip())
    soft_traction_return_codes.append(ret_code)
    asp_rules_coverage = len(normal_indices) / total_number_of_states
    log.info(f"ASP rules coverage: {asp_rules_coverage}")
    log.info("======================================")

    return {
        "normal_indices": normal_indices,
        "asp_rules": rules,
        "return_codes": soft_traction_return_codes,
        "asp_coverage": asp_rules_coverage,
        "policy_error_cmp_to_q": threshold_log["policy_error_cmp_to_q"],
    }


def post_train_eval(eval_cfg: DictConfig) -> dict[str, Any]:
    experiment_name = f"{eval_cfg['experiment_name']}"
    use_ndnf = "ndnf" in experiment_name

    assert use_ndnf and not eval_cfg["use_eo"] and eval_cfg["use_mt"]

    target_policy_csv_path = Path(eval_cfg["target_policy_csv_path"])
    if not target_policy_csv_path.exists():
        raise FileNotFoundError(
            f"The target policy csv file {target_policy_csv_path} does not exist!"
        )

    ret_dict: dict[int, list[int]] = dict(
        [(c.value, []) for c in ToyTextSoftExtractionReturnCode]
    )

    asp_coverage_list: list[float] = []
    policy_error_cmp_to_q_list: list[float] = []

    json_dict = {}

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
        ret = post_training(model, target_policy_csv_path, eval_cfg, model_dir)

        # Use the random seed as the identifier and add to the corresponding
        # list
        if isinstance(ret, dict):
            # Finished run
            ret_dict[
                ToyTextSoftExtractionReturnCode.SOFT_EXTRACTION_FINISH
            ].append(s)
            asp_coverage_list.append(ret["asp_coverage"])
            policy_error_cmp_to_q_list.append(ret["policy_error_cmp_to_q"])
            json_dict[s] = ret
        else:
            ret_dict[ret].append(s)
            json_dict[s] = {"failure_code": ret.value}
        log.info("======================================\n")

    for k, v in ret_dict.items():
        if k == ToyTextSoftExtractionReturnCode.SOFT_EXTRACTION_FINISH:
            log.info(f"Finished: {sorted(v)}")
        else:
            log.info(f"{ToyTextSoftExtractionReturnCode(k).name}: {sorted(v)}")

    log.info(f"Average asp coverage: {np.array(asp_coverage_list).mean()}")
    log.info(
        f"Average policy error compared to Q: {np.array(policy_error_cmp_to_q_list).mean()}"
    )

    with open("eval.json", "w") as f:
        json.dump(json_dict, f, indent=4)

    return {
        "total_finish_runs": len(
            ret_dict[ToyTextSoftExtractionReturnCode.SOFT_EXTRACTION_FINISH]
        ),
        "avg_asp_coverage": np.array(asp_coverage_list).mean(),
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
            msg_body += f"Total success runs: {ret_dict['total_finish_runs']}\n"
            msg_body += f"Average win rate: {ret_dict['avg_asp_coverage']}\n"
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
