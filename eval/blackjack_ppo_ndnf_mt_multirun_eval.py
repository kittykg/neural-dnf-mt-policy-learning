# This script evaluates the NDNF MT model on the Blackjack environment compared
# to a target Q-value table.
from collections import OrderedDict
from copy import deepcopy
import json
import logging
from pathlib import Path
import random
import sys
import traceback
from typing import Any

import clingo
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import hydra
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


from neural_dnf.post_training import (
    prune_neural_dnf,
    apply_threshold,
    extract_asp_rules,
    get_thresholding_upper_bound,
)

from blackjack_common import (
    decode_tuple_obs,
    get_target_policy,
    create_policy_plots_from_asp,
    create_policy_plots_from_action_distribution,
    TargetPolicyType,
)
from blackjack_ppo import (
    construct_model,
    construct_single_environment,
    get_agent_policy,
    BlackjackPPONDNFMutexTanhAgent,
)
from eval.common import ToyTextEnvFailureCode
from utils import post_to_discord_webhook


DEFAULT_GEN_SEED = 2
BLACKJACK_SINGLE_ENV_NUM_EPISODES = 500
DEVICE = torch.device("cpu")
BASE_STORAGE_DIR = root / "blackjack_ppo_storage"
log = logging.getLogger()
single_env = construct_single_environment()


def get_ndnf_action(
    model: BlackjackPPONDNFMutexTanhAgent,
    obs: dict[str, Tensor],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    # Use normal tanh interpretation
    with torch.no_grad():
        actions = model.get_actions(
            preprocessed_obs=obs,
            use_argmax=True,
        )
    return actions


def simulate_fn(
    model: BlackjackPPONDNFMutexTanhAgent,
    target_policy: TargetPolicyType,
) -> dict[str, Any]:
    logs: dict[str, Any] = {
        "mutual_exclusivity": True,
        "missing_actions": False,
    }

    obs_list = [obs for obs in target_policy.keys()]
    target_q_actions = np.array([target_policy[obs] for obs in obs_list])
    decode_input_nd_array = np.stack(
        [decode_tuple_obs(obs) for obs in obs_list]
    )
    decode_input_nd_array = np.where(
        decode_input_nd_array == 0, -1, decode_input_nd_array
    )

    actions, tanh_actions = get_ndnf_action(
        model,
        {
            "decode_input": torch.tensor(
                decode_input_nd_array, dtype=torch.float32, device=DEVICE
            )
        },
    )
    tanh_actions_discretised = np.count_nonzero(tanh_actions > 0, axis=1)
    if np.any(tanh_actions_discretised > 1):
        logs["mutual_exclusivity"] = False
    if np.any(tanh_actions_discretised == 0):
        logs["missing_actions"] = True

    policy_error_cmp_to_q = np.count_nonzero(actions != target_q_actions) / len(
        target_q_actions
    )
    logs["policy_error_cmp_to_q"] = policy_error_cmp_to_q
    logs["actions"] = actions

    return logs


def post_training(
    model: BlackjackPPONDNFMutexTanhAgent,
    target_policy: TargetPolicyType,
    eval_cfg: DictConfig,
    model_dir: Path,
) -> ToyTextEnvFailureCode | dict[str, Any]:

    def _simulate_with_print(model_name: str) -> dict[str, Any]:
        logs = simulate_fn(model, target_policy)
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
            <= t_a
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

    # Stage 1: Evaluate the model post-training
    og_ndnf_mt_logs = _simulate_with_print("NDNF MT")
    if og_ndnf_mt_logs["missing_actions"]:
        log.info("NDNF MT has missing actions!")
        return ToyTextEnvFailureCode.FAIL_AT_EVAL_NDNF_MT_MISS_ACTION
    if not og_ndnf_mt_logs["mutual_exclusivity"]:
        log.info("NDNF MT is not mutually exclusive!")
        return ToyTextEnvFailureCode.FAIL_AT_EVAL_NDNF_MT_NOT_ME

    action_distribution = get_agent_policy(
        model, target_policy, torch.device("cpu")
    )
    plot = create_policy_plots_from_action_distribution(
        target_policy,
        action_distribution,
        model_dir.name,
        argmax=True,
        plot_diff=True,
    )
    plot.savefig(f"{model_dir.name}_policy.png")
    plt.close()

    log.info("======================================")

    # Stage 2: Prune the model

    def prune_model() -> ToyTextEnvFailureCode | None:
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
            log.info(f"Pruning iteration: {prune_count+1}")
            prune_result_dict = prune_neural_dnf(
                model.actor,
                simulate_fn,
                {
                    "model": model,
                    "target_policy": target_policy,
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
                f"NDNF MT dis - (Prune iteration: {prune_count})"
            )

            if post_prune_ndnf_mt_log["missing_actions"]:
                log.info("Post prune NDNF MT has missing actions!")
                return ToyTextEnvFailureCode.FAIL_AT_PRUNE_MISS_ACTION
            if not post_prune_ndnf_mt_log["mutual_exclusivity"]:
                log.info("Post prune NDNF MT is not mutually exclusive!")
                return ToyTextEnvFailureCode.FAIL_AT_PRUNE_NOT_ME

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
        if ret is not None and isinstance(ret, ToyTextEnvFailureCode):
            return ret
        torch.save(model.state_dict(), model_dir / "model_mr_pruned.pth")

    _simulate_with_print("NDNF MT pruned")

    action_distribution = get_agent_policy(
        model, target_policy, torch.device("cpu")
    )
    plot = create_policy_plots_from_action_distribution(
        target_policy,
        action_distribution,
        f"{model_dir.name}_pruned",
        argmax=True,
        plot_diff=True,
    )
    plot.savefig(f"{model_dir.name}_pruned_policy.png")
    plt.close()

    log.info("======================================")

    # 3. Thresholding
    og_conj_weight = model.actor.conjunctions.weights.data.clone()
    og_disj_weight = model.actor.disjunctions.weights.data.clone()

    def threshold_model() -> ToyTextEnvFailureCode | list[float]:
        log.info("Thresholding the model...")

        threshold_upper_bound = get_thresholding_upper_bound(model.actor)
        log.info(f"Threshold upper bound: {threshold_upper_bound}")

        t_vals = torch.arange(0, threshold_upper_bound, 0.01)
        result_dicts = []

        for v in t_vals:
            apply_threshold(model.actor, og_conj_weight, og_disj_weight, v)
            r = simulate_fn(model, target_policy)
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

        if len(t_vals_candidates) == 0:
            log.info("No thresholding candidate found!")
            return ToyTextEnvFailureCode.FAIL_AT_THRESHOLD_NO_CANDIDATE

        log.info(
            f"t_vals_candidates: {[round(v, 2) for v in t_vals_candidates]}"
        )
        return t_vals_candidates

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

    if (model_dir / "thresholded_model.pth").exists():
        thresholded_state = torch.load(
            model_dir / "thresholded_model.pth", map_location=DEVICE
        )
        model.load_state_dict(thresholded_state)
    elif (model_dir / "threshold_val_candidates.json").exists():
        with open(model_dir / "threshold_val_candidates.json", "r") as f:
            threshold_json_dict = json.load(f)
            if not threshold_json_dict["threshold_success"]:
                log.info(
                    "Thresholding failed with no candidate in the previous run!"
                )
                return ToyTextEnvFailureCode.FAIL_AT_THRESHOLD_NO_CANDIDATE
            t_vals_candidates = threshold_json_dict["threshold_vals"]
        apply_threshold_with_candidate_list(t_vals_candidates)
    else:
        ret = threshold_model()
        threshold_json_dict = {}
        with open(model_dir / "threshold_val_candidates.json", "w") as f:
            if isinstance(ret, ToyTextEnvFailureCode):
                threshold_json_dict["threshold_success"] = False
                json.dump(threshold_json_dict, f)
                return ret
            t_vals_candidates = ret
            threshold_json_dict["threshold_success"] = True
            threshold_json_dict["threshold_vals"] = t_vals_candidates
            json.dump(threshold_json_dict, f)
        apply_threshold_with_candidate_list(t_vals_candidates)

    _simulate_with_print("NDNF MT (thresholded)")

    action_distribution = get_agent_policy(
        model, target_policy, torch.device("cpu")
    )
    plot = create_policy_plots_from_action_distribution(
        target_policy,
        action_distribution,
        f"{model_dir.name}_thresholded",
        argmax=True,
        plot_diff=True,
    )
    plot.savefig(f"{model_dir.name}_thresholded_policy.png")
    plt.close()

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
        log.info(r)

    log.info("======================================")

    # 5. Evaluate Rule
    res_list = []

    for i in range(BLACKJACK_SINGLE_ENV_NUM_EPISODES):
        obs, _ = single_env.reset()

        terminated = False
        truncated = False
        reward_sum = 0

        while not terminated and not truncated:
            input = [
                f"a_{i}."
                for i, a in enumerate(decode_tuple_obs(obs))
                if a.item() != 0
            ]

            ctl = clingo.Control(["--warn=none"])
            show_statements = [f"#show disj_{i}/0." for i in range(2)]
            ctl.add("base", [], " ".join(input + show_statements + rules))
            ctl.ground([("base", [])])
            with ctl.solve(yield_=True) as handle:  # type: ignore
                all_answer_sets = [str(a) for a in handle]

            if len(all_answer_sets) != 1:
                # No model or multiple answer sets, should not happen
                log.info(
                    f"No model or multiple answer sets when evaluating rules."
                )
                return (
                    ToyTextEnvFailureCode.FAIL_AT_RULE_EVAL_NOT_ONE_ANSWER_SET
                )

            if all_answer_sets[0] == "":
                log.info(f"No output action!")
                return ToyTextEnvFailureCode.FAIL_AT_RULE_EVAL_MISSING_ACTION

            output_classes = all_answer_sets[0].split(" ")
            if len(output_classes) == 0:
                log.info(f"No output action!")
                return ToyTextEnvFailureCode.FAIL_AT_RULE_EVAL_MISSING_ACTION
            output_classes_set = set([int(o[5:]) for o in output_classes])

            if len(output_classes_set) != 1:
                log.info(
                    f"Output set: {output_classes_set} not exactly one item!"
                )
                return (
                    ToyTextEnvFailureCode.FAIL_AT_RULE_EVAL_MORE_THAN_ONE_ACTION
                )

            action = list(output_classes_set)[0]
            obs, reward, terminated, truncated, _ = single_env.step(action)
            reward_sum += reward  # type: ignore

        res_list.append(reward_sum)

    num_wins = np.sum(np.array(res_list) == 1)
    num_losses = np.sum(np.array(res_list) == -1)
    num_draws = np.sum(np.array(res_list) == 0)

    log.info(f"Average reward: {np.mean(res_list)}")
    log.info(
        f"Number of wins: {num_wins}\tPercentage: {num_wins / len(res_list)}"
    )
    log.info(
        f"Number of losses: {num_losses}\tPercentage: {num_losses / len(res_list)}"
    )
    log.info(
        f"Number of draws: {num_draws}\tPercentage: {num_draws / len(res_list)}"
    )

    asp_policy: TargetPolicyType = OrderedDict()
    for obs in target_policy.keys():
        input = [
            f"a_{i}."
            for i, a in enumerate(decode_tuple_obs(obs))
            if a.item() != 0
        ]

        ctl = clingo.Control(["--warn=none"])
        show_statements = [f"#show disj_{i}/0." for i in range(2)]
        ctl.add("base", [], " ".join(input + show_statements + rules))
        ctl.ground([("base", [])])
        with ctl.solve(yield_=True) as handle:  # type: ignore
            all_answer_sets = [str(a) for a in handle]

        if len(all_answer_sets) != 1:
            # No model or multiple answer sets, should not happen
            log.info(f"No model or multiple answer sets when evaluating rules.")
            return ToyTextEnvFailureCode.FAIL_AT_RULE_EVAL_NOT_ONE_ANSWER_SET

        if all_answer_sets[0] == "":
            log.info(f"No output action!")
            return ToyTextEnvFailureCode.FAIL_AT_RULE_EVAL_MISSING_ACTION

        output_classes = all_answer_sets[0].split(" ")
        if len(output_classes) == 0:
            log.info(f"No output action!")
            return ToyTextEnvFailureCode.FAIL_AT_RULE_EVAL_MISSING_ACTION
        output_classes_set = set([int(o[5:]) for o in output_classes])

        if len(output_classes_set) != 1:
            log.info(f"Output set: {output_classes_set} not exactly one item!")
            return ToyTextEnvFailureCode.FAIL_AT_RULE_EVAL_MORE_THAN_ONE_ACTION

        action = list(output_classes_set)[0]
        asp_policy[obs] = action

    obs_list = [obs for obs in target_policy.keys()]
    asp_policy_error_cmp_q_table = np.count_nonzero(
        np.array([asp_policy[obs] for obs in obs_list])
        != np.array([target_policy[obs] for obs in obs_list])
    ) / len(target_policy)

    return {
        "asp_win_rate": num_wins / len(res_list),
        "asp_rules": rules,
        "asp_policy_error_cmp_q_table": asp_policy_error_cmp_q_table,
        "asp_policy": asp_policy,
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
    target_policy = get_target_policy(target_policy_csv_path)

    ret_dict: dict[int, list[int]] = dict(
        [(c.value, []) for c in ToyTextEnvFailureCode] + [(0, [])]
    )
    win_rate_list: list[float] = []
    json_dict = {}

    for s in eval_cfg["multirun_seeds"]:
        # Load agent
        model_dir = BASE_STORAGE_DIR / f"{experiment_name}_{s}"
        model: BlackjackPPONDNFMutexTanhAgent = construct_model(
            num_latent=eval_cfg["model_latent_size"],
            use_ndnf=use_ndnf,
            use_decode_obs=True,
            use_eo=False,
            use_mt=True,
        )  # type: ignore
        model.to(DEVICE)
        model_state = torch.load(model_dir / "model.pth", map_location=DEVICE)
        model.load_state_dict(model_state)
        model.eval()

        log.info(f"Experiment {model_dir.name} loaded!")
        ret = post_training(model, target_policy, eval_cfg, model_dir)

        # Use the random seed as the identifier and add to the corresponding
        # list
        if isinstance(ret, dict):
            # Successful run
            ret_dict[0].append(s)
            win_rate_list.append(ret["asp_win_rate"])

            asp_policy = ret.pop("asp_policy")
            plot = create_policy_plots_from_asp(
                target_policy,
                asp_policy,
                f"{experiment_name}_{s}",
                plot_diff=True,
            )
            plot.savefig(f"{experiment_name}_{s}_asp_policy.png")
            plt.close()

            json_dict[s] = ret
        else:
            ret_dict[ret].append(s)
            json_dict[s] = {"failure_code": ret.value}
        log.info("======================================\n")

    for k, v in ret_dict.items():
        if k == 0:
            log.info(f"Success: {sorted(v)}")
        else:
            log.info(f"{ToyTextEnvFailureCode(k).name}: {sorted(v)}")

    log.info(f"Win rate list: {win_rate_list}")
    log.info(f"Average win rate: {np.array(win_rate_list).mean()}")
    log.info(f"Win rate std: {np.array(win_rate_list).std()}")

    with open("eval.json", "w") as f:
        json.dump(json_dict, f, indent=4)

    return {
        "total_success_runs": len(win_rate_list),
        "avg_win_rate": np.array(win_rate_list).mean(),
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
        avg_win_rate = post_train_eval(eval_cfg)
        if use_discord_webhook:
            msg_body = f"Success!\n"
            msg_body += (
                f"Total success runs: {avg_win_rate['total_success_runs']}\n"
            )
            msg_body += f"Average win rate: {avg_win_rate['avg_win_rate']}"
    except BaseException as e:
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
    import multiprocessing as mp

    if mp.get_start_method() != "fork":
        mp.set_start_method("fork", force=True)

    run_eval()
