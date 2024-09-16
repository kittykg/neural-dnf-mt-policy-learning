# This script evaluates the NDNF MT agent on the DoorCorridor environment
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


from corridor_grid.envs import DoorCorridorEnv
from neural_dnf.post_training import (
    prune_neural_dnf,
    apply_threshold,
    extract_asp_rules,
    get_thresholding_upper_bound,
)
from common import synthesize
from door_corridor_ppo import construct_model, make_env, DCPPONDNFMutexTanhAgent
from eval.asp_inference_common import (
    ASPRuleEvaluationFailureCode,
    evaluate_rule_on_env,
)
from eval.common import DoorCorridorFailureCode
from utils import post_to_discord_webhook


BASE_STORAGE_DIR = root / "dc_ppo_storage"
DEFAULT_GEN_SEED = 2
DEVICE = torch.device("cpu")
NUM_EPISODES = 100
NUM_PROCESSES = 8

FIRST_PRUNE_MODEL_PTH_NAME = "model_mr_pruned.pth"
THRESHOLD_MODEL_PTH_NAME = "thresholded_model.pth"
THRESHOLD_JSON_NAME = "threshold_val_candidates.json"
SECOND_PRUNE_MODEL_PTH_NAME = "model_2nd_mr_pruned.pth"
ASP_RULES_FILE_NAME = "asp_rules.lp"


envs = gym.vector.SyncVectorEnv(
    [make_env(i, i, False) for i in range(NUM_PROCESSES)]
)
single_env = DoorCorridorEnv(render_mode="rgb_array")
log = logging.getLogger()


def get_ndnf_action(
    model: DCPPONDNFMutexTanhAgent,
    discretise_img_encoding: bool,
    obs: dict[str, Tensor],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    # Use normal tanh interpretation
    with torch.no_grad():
        actions = model.get_actions(
            preprocessed_obs=obs,
            use_argmax=True,
            discretise_img_encoding=discretise_img_encoding,
        )
    return actions


def simulate_fn(
    envs: gym.vector.SyncVectorEnv,
    action_fn: Callable[[dict[str, Tensor]], tuple],
    num_episodes=NUM_EPISODES,
) -> dict[str, Any]:
    logs = {
        "num_frames_per_episode": [],
        "return_per_episode": [],
        "mutual_exclusivity": True,
        "missing_actions": False,
    }
    next_obs_dict, _ = envs.reset()
    next_obs = torch.Tensor(next_obs_dict["image"]).to(DEVICE)
    next_obs_dict = {"image": next_obs}

    log_done_counter = 0
    log_episode_return = torch.zeros(NUM_PROCESSES, device=DEVICE)
    log_episode_num_frames = torch.zeros(NUM_PROCESSES, device=DEVICE)

    while log_done_counter < num_episodes:
        actions = action_fn(next_obs_dict)
        # For NDNF based model, the get_actions() returns a tuple of
        # actions and tanh interpretation. We check the if the tanh
        # interpretation is greater than 0.
        tanh_action = np.count_nonzero(actions[1] > 0, axis=1)
        if np.any(tanh_action > 1):
            logs["mutual_exclusivity"] = False
        if np.any(tanh_action == 0):
            logs["missing_actions"] = True
        next_obs_dict, reward, terminations, truncations, _ = envs.step(
            actions[0]
        )
        next_obs = torch.Tensor(next_obs_dict["image"]).to(DEVICE)
        next_obs_dict = {"image": next_obs}
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


def post_training(model_dir: Path, model: DCPPONDNFMutexTanhAgent) -> int:
    simulate = lambda action_fn: simulate_fn(envs, action_fn)

    def _simulate_with_print(action_fn, model_name: str) -> dict[str, Any]:
        logs = simulate(action_fn)

        num_frames = sum(logs["num_frames_per_episode"])
        return_per_episode = synthesize(logs["return_per_episode"])
        num_frames_per_episode = synthesize(logs["num_frames_per_episode"])

        log.info(
            "{}\tF {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}".format(
                model_name,
                num_frames,
                *return_per_episode.values(),
                *num_frames_per_episode.values(),
            )
        )
        log.info(f"Mutual exclusivity: {logs['mutual_exclusivity']}")
        log.info(f"Missing actions: {logs['missing_actions']}")
        return logs

    def is_truncated(logs: dict[str, Any]) -> bool:
        return (
            abs(np.array(logs["return_per_episode"]).mean())
            >= single_env.max_steps
        )

    _ndnf_mt_action_fn = lambda obs: get_ndnf_action(model, False, obs)
    _ndnf_mt_dis_action_fn = lambda obs: get_ndnf_action(model, True, obs)

    # Stage 1: Evaluate the model post-training
    og_ndnf_mt_logs = _simulate_with_print(_ndnf_mt_action_fn, "NDNF MT")
    # Discretise image encoding output
    og_ndnf_mt_dis_logs = _simulate_with_print(
        _ndnf_mt_dis_action_fn, "NDNF MT dis"
    )
    if is_truncated(og_ndnf_mt_logs):
        log.info("NDNF MT is not finishing the environment!")
        return DoorCorridorFailureCode.FAIL_AT_EVAL_NDNF_MT_TRUNCATED
    if not is_truncated(og_ndnf_mt_logs) and is_truncated(og_ndnf_mt_dis_logs):
        log.info("NDNF MT dis is truncated after discretisation!")
        return DoorCorridorFailureCode.FAIL_AT_EVAL_NDNF_MT_DIS_TRUNCATED
    if og_ndnf_mt_dis_logs["missing_actions"]:
        log.info("NDNF MT dis has missing actions!")
        return DoorCorridorFailureCode.FAIL_AT_EVAL_NDNF_MT_DIS_MISS_ACTION
    if not og_ndnf_mt_dis_logs["mutual_exclusivity"]:
        log.info("NDNF MT dis is not mutually exclusive!")
        return DoorCorridorFailureCode.FAIL_AT_EVAL_NDNF_MT_DIS_NOT_ME

    log.info("======================================")

    # Stage 2: Prune the model
    def prune_model() -> DoorCorridorFailureCode | None:
        log.info("Pruning the model...")

        def comparison_fn(og_log, new_log):
            if not new_log["mutual_exclusivity"]:
                return False
            if new_log["missing_actions"]:
                return False
            return (
                np.array(og_log["return_per_episode"]).mean()
                - np.array(new_log["return_per_episode"]).mean()
                <= 0
            )

        sd_list = []
        prune_count = 0

        while True:
            log.info(f"Pruning iteration: {prune_count+1}")
            prune_result_dict = prune_neural_dnf(
                model.actor,
                lambda: simulate(_ndnf_mt_dis_action_fn),
                {},
                comparison_fn,
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

            post_prune_ndnf_mt_dis_log = _simulate_with_print(
                _ndnf_mt_dis_action_fn,
                f"NDNF MT dis - (Prune iteration: {prune_count})",
            )

            if is_truncated(post_prune_ndnf_mt_dis_log):
                log.info("Post prune NDNF MT dis losses performance!")
                return DoorCorridorFailureCode.FAIL_AT_PRUNE_TRUNCATED
            if post_prune_ndnf_mt_dis_log["missing_actions"]:
                log.info("Post prune NDNF MT dis has missing actions!")
                return DoorCorridorFailureCode.FAIL_AT_PRUNE_MISS_ACTION
            if not post_prune_ndnf_mt_dis_log["mutual_exclusivity"]:
                log.info("Post prune NDNF MT dis is not mutually exclusive!")
                return DoorCorridorFailureCode.FAIL_AT_PRUNE_NOT_ME

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
        ret = prune_model()
        if ret is not None and isinstance(ret, DoorCorridorFailureCode):
            return ret
        torch.save(model.state_dict(), model_dir / FIRST_PRUNE_MODEL_PTH_NAME)

    _simulate_with_print(_ndnf_mt_dis_action_fn, f"NDNF MT dis pruned")

    log.info("======================================")

    # 3. Thresholding
    og_conj_weight = model.actor.conjunctions.weights.data.clone()
    og_disj_weight = model.actor.disjunctions.weights.data.clone()

    def threshold_model() -> DoorCorridorFailureCode | list[float]:
        log.info("Thresholding the model...")

        threshold_upper_bound = get_thresholding_upper_bound(model.actor)
        log.info(f"Threshold upper bound: {threshold_upper_bound}")

        t_vals = torch.arange(0, threshold_upper_bound, 0.01)
        result_dicts = []
        for v in t_vals:
            apply_threshold(model.actor, og_conj_weight, og_disj_weight, v)
            result_dicts.append(simulate(_ndnf_mt_dis_action_fn))

        t_vals_candidates = []
        for i, d in enumerate(result_dicts):
            if (
                (
                    np.array(d["return_per_episode"]).mean()
                    < np.array(og_ndnf_mt_dis_logs["return_per_episode"]).mean()
                )
                or d["missing_actions"]
                or not d["mutual_exclusivity"]
            ):
                continue
            t_vals_candidates.append(t_vals[i].item())
        if len(t_vals_candidates) == 0:
            log.info("No thresholding candidate found!")
            return DoorCorridorFailureCode.FAIL_AT_THRESHOLD_NO_CANDIDATE

        log.info(f"t_vals_candidates: {t_vals_candidates}")
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
        torch.save(model.state_dict(), model_dir / THRESHOLD_MODEL_PTH_NAME)

    if (model_dir / THRESHOLD_MODEL_PTH_NAME).exists():
        thresholded_state = torch.load(
            model_dir / THRESHOLD_MODEL_PTH_NAME, map_location=DEVICE
        )
        model.load_state_dict(thresholded_state)
    elif (model_dir / THRESHOLD_JSON_NAME).exists():
        with open(model_dir / THRESHOLD_JSON_NAME, "r") as f:
            threshold_json_dict = json.load(f)
            if not threshold_json_dict["threshold_success"]:
                log.info(
                    "Thresholding failed with no candidate in the previous run!"
                )
                return DoorCorridorFailureCode.FAIL_AT_THRESHOLD_NO_CANDIDATE
            t_vals_candidates = threshold_json_dict["threshold_vals"]
        apply_threshold_with_candidate_list(t_vals_candidates)
    else:
        ret = threshold_model()
        threshold_json_dict = {}
        with open(model_dir / THRESHOLD_JSON_NAME, "w") as f:
            if isinstance(ret, DoorCorridorFailureCode):
                threshold_json_dict["threshold_success"] = False
                json.dump(threshold_json_dict, f)
                return ret
            t_vals_candidates = ret
            threshold_json_dict["threshold_success"] = True
            threshold_json_dict["threshold_vals"] = t_vals_candidates
            json.dump(threshold_json_dict, f)
        apply_threshold_with_candidate_list(t_vals_candidates)

    _simulate_with_print(_ndnf_mt_dis_action_fn, "NDNF MT dis (thresholded)")

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
        ret = prune_model()
        if ret is not None and isinstance(ret, DoorCorridorFailureCode):
            return ret
        torch.save(model.state_dict(), model_dir / SECOND_PRUNE_MODEL_PTH_NAME)

    _simulate_with_print(_ndnf_mt_dis_action_fn, f"NDNF MT dis 2nd prune")

    log.info("======================================")

    # 5. Rule extraction
    if (model_dir / ASP_RULES_FILE_NAME).exists():
        with open(model_dir / ASP_RULES_FILE_NAME, "r") as f:
            rules = f.readlines()
    else:
        rules: list[str] = extract_asp_rules(model.actor.state_dict())  # type: ignore
        with open(model_dir / ASP_RULES_FILE_NAME, "w") as f:
            f.write("\n".join(rules))
    for r in rules:
        log.info(r)

    log.info("======================================")

    # 6. Evaluate Rule
    def context_generation(obs: dict[str, np.ndarray]) -> list[str]:
        with torch.no_grad():
            raw_img_encoding = model.get_img_encoding(
                preprocessed_obs={
                    "image": torch.tensor(obs["image"].copy(), device=DEVICE)
                    .unsqueeze(0)
                    .float()
                }
            ).squeeze(0)
        img_encoding = [
            f"a_{a.item()}." for a in torch.nonzero(raw_img_encoding > 0)
        ]
        log.info(img_encoding)
        return img_encoding

    ret = evaluate_rule_on_env(
        env=single_env,
        context_encoding_generation_fn=context_generation,
        num_actions=DoorCorridorEnv.get_num_actions(),
        rules=rules,
        eval_num_episodes=1,
        do_logging=True,
    )
    if isinstance(ret, ASPRuleEvaluationFailureCode):
        return {
            ASPRuleEvaluationFailureCode.FAIL_AT_RULE_EVAL_NOT_ONE_ANSWER_SET: DoorCorridorFailureCode.FAIL_AT_RULE_EVAL_NOT_ONE_ANSWER_SET,
            ASPRuleEvaluationFailureCode.FAIL_AT_RULE_EVAL_MISSING_ACTION: DoorCorridorFailureCode.FAIL_AT_RULE_EVAL_MISSING_ACTION,
            ASPRuleEvaluationFailureCode.FAIL_AT_RULE_EVAL_MORE_THAN_ONE_ACTION: DoorCorridorFailureCode.FAIL_AT_RULE_EVAL_MORE_THAN_ONE_ACTION,
            ASPRuleEvaluationFailureCode.FAIL_AT_RULE_EVAL_TRUNCATED: DoorCorridorFailureCode.FAIL_AT_RULE_EVAL_TRUNCATED,
        }[ret]

    return 0


def post_train_eval(eval_cfg: DictConfig):
    experiment_name = f"{eval_cfg['experiment_name']}"
    use_ndnf = "ndnf" in experiment_name

    assert use_ndnf and not eval_cfg["use_eo"] and eval_cfg["use_mt"]

    ret_dict: dict[int, list[int]] = dict(
        [(c.value, []) for c in DoorCorridorFailureCode] + [(0, [])]
    )

    for s in eval_cfg["multirun_seeds"]:
        # Load agent
        model_dir = BASE_STORAGE_DIR / f"{experiment_name}_{s}"
        model: DCPPONDNFMutexTanhAgent = construct_model(
            eval_cfg,
            DoorCorridorEnv.get_num_actions(),
            use_ndnf,
            single_env.observation_space["image"],  # type: ignore
        )
        model.to(DEVICE)
        model_state = torch.load(model_dir / "model.pth", map_location=DEVICE)
        model.load_state_dict(model_state)
        model.eval()

        log.info(f"Experiment {model_dir.name} loaded!")
        ret_code = post_training(model_dir, model)
        ret_dict[ret_code].append(s)  # use the random seed as the identifier
        log.info("======================================\n")

    for k, v in ret_dict.items():
        if k == 0:
            log.info(f"Success: {sorted(v)}")
        else:
            log.info(f"{DoorCorridorFailureCode(k).name}: {sorted(v)}")


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
    torch.set_warn_always(False)

    import multiprocessing as mp

    if mp.get_start_method() != "forkserver":
        mp.set_start_method("forkserver", force=True)

    run_eval()
