# This script evaluates the NDNF based agent on MDP SpecialStateCorridor envs
# i.e. using state number as observation of agent
# There are deterministic policies for each envs, so the agent should be able to
# learn and extract ASP rules
from copy import deepcopy
import json
import logging
from pathlib import Path
import random
import sys
import traceback
from typing import Any, Callable

import clingo
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


from corridor_grid.envs.base_ss_corridor import BaseSpecialStateCorridorEnv
from neural_dnf.post_training import (
    prune_neural_dnf,
    apply_threshold,
    extract_asp_rules,
    get_thresholding_upper_bound,
)
from common import synthesize
from ss_corridor_ppo import (
    construct_model,
    construct_single_environment,
    make_env,
    ss_corridor_preprocess_obs,
    SSCPPONDNFBasedAgent,
    SSCPPONDNFEOAgent,
    SSCPPONDNFMutexTanhAgent,
)
from eval.common import SpecialStateCorridorFailureCode
from utils import post_to_discord_webhook


DEFAULT_GEN_SEED = 2
NUM_PROCESSES = 8
NUM_EPISODES = 100
DEVICE = torch.device("cpu")
BASE_STORAGE_DIR = root / "ssc_ppo_storage"

log = logging.getLogger()


def get_ndnf_action(
    model: SSCPPONDNFBasedAgent,
    obs: dict[str, Tensor],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    with torch.no_grad():
        actions = model.get_actions(
            preprocessed_obs=obs,
            use_argmax=True,
        )
    return actions


def simulate_fn(
    envs: gym.vector.SyncVectorEnv,
    action_fn: Callable[[dict[str, Tensor]], tuple],
    process_obs=Callable[[dict], Tensor],
    num_episodes=NUM_EPISODES,
) -> dict[str, Any]:
    logs = {
        "num_frames_per_episode": [],
        "return_per_episode": [],
        "mutual_exclusivity": True,
        "missing_actions": False,
    }
    next_obs_dict, _ = envs.reset()
    next_obs = process_obs(next_obs_dict)
    next_obs_dict = {"input": next_obs}

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


def post_training(
    model_dir: Path,
    model: SSCPPONDNFBasedAgent,
    envs: gym.vector.SyncVectorEnv,
    single_env: BaseSpecialStateCorridorEnv,
    process_obs: Callable[[dict], Tensor],
    env_max_steps: int,
):
    simulate = lambda action_fn: simulate_fn(envs, action_fn, process_obs)

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
        return abs(np.array(logs["return_per_episode"]).mean()) >= env_max_steps

    _action_fn = lambda obs: get_ndnf_action(model, obs)

    if isinstance(model, SSCPPONDNFEOAgent):
        base_stage_name = "NDNF EO"
        plain_ndnf_model = model.to_ndnf_agent()
        _plain_ndnf_action_fn = lambda obs: get_ndnf_action(
            plain_ndnf_model, obs
        )
    elif isinstance(model, SSCPPONDNFMutexTanhAgent):
        base_stage_name = "NDNF MT"
    else:
        base_stage_name = "NDNF"

    # Stage 1: Evaluate the model post-training
    og_logs = _simulate_with_print(_action_fn, base_stage_name)

    if is_truncated(og_logs):
        log.info(f"{base_stage_name} is not finishing the environment!")
        if isinstance(model, SSCPPONDNFMutexTanhAgent):
            return (
                SpecialStateCorridorFailureCode.FAIL_AT_EVAL_NDNF_MT_TRUNCATED
            )
        elif isinstance(model, SSCPPONDNFEOAgent):
            return (
                SpecialStateCorridorFailureCode.FAIL_AT_EVAL_NDNF_EO_TRUNCATED
            )
        return SpecialStateCorridorFailureCode.FAIL_AT_EVAL_NDNF_TRUNCATED

    if isinstance(model, SSCPPONDNFEOAgent):
        # If the model is EO, we remove the EO and evaluate the model
        og_plain_ndnf_logs = _simulate_with_print(
            _plain_ndnf_action_fn, "Plain NDNF (EO removed)"
        )
        if is_truncated(og_plain_ndnf_logs):
            log.info(
                "Plain NDNF (EO removed) is not finishing the environment!"
            )
            return (
                SpecialStateCorridorFailureCode.FAIL_AT_EVAL_NDNF_LOSS_PERFORMANCE_AFTER_EO_REMOVED
            )

        # If the model is not truncated and before proceeding to the next stage,
        # we set model to the plain NDNF model
        model = plain_ndnf_model
        base_stage_name = "Plain NDNF (EO removed)"
        _action_fn = lambda obs: get_ndnf_action(model, obs)

    log.info("======================================")

    # Stage 2: Prune the model
    def prune_model() -> SpecialStateCorridorFailureCode | None:
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
                lambda: simulate(_action_fn),
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
                _action_fn,
                f"{base_stage_name} - (Prune iteration: {prune_count})",
            )

            if is_truncated(post_prune_ndnf_mt_dis_log):
                log.info(f"Post prune {base_stage_name} losses performance!")
                return SpecialStateCorridorFailureCode.FAIL_AT_PRUNE_TRUNCATED
            if post_prune_ndnf_mt_dis_log["missing_actions"]:
                log.info(f"Post prune {base_stage_name} has missing actions!")
                return SpecialStateCorridorFailureCode.FAIL_AT_PRUNE_MISS_ACTION
            if not post_prune_ndnf_mt_dis_log["mutual_exclusivity"]:
                log.info(
                    f"Post prune {base_stage_name} is not mutually exclusive!"
                )
                return SpecialStateCorridorFailureCode.FAIL_AT_PRUNE_NOT_ME

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
        if ret is not None and isinstance(ret, SpecialStateCorridorFailureCode):
            return ret
        torch.save(model.state_dict(), model_dir / "model_mr_pruned.pth")

    _simulate_with_print(_action_fn, f"{base_stage_name} pruned")

    log.info("======================================")

    # 3. Thresholding
    og_conj_weight = model.actor.conjunctions.weights.data.clone()
    og_disj_weight = model.actor.disjunctions.weights.data.clone()

    def threshold_model() -> SpecialStateCorridorFailureCode | list[float]:
        log.info("Thresholding the model...")

        threshold_upper_bound = get_thresholding_upper_bound(model.actor)
        log.info(f"Threshold upper bound: {threshold_upper_bound}")

        t_vals = torch.arange(0, threshold_upper_bound, 0.01)
        result_dicts = []
        for v in t_vals:
            apply_threshold(model.actor, og_conj_weight, og_disj_weight, v)
            result_dicts.append(simulate(_action_fn))

        t_vals_candidates = []
        for i, d in enumerate(result_dicts):
            if (
                (
                    np.array(d["return_per_episode"]).mean()
                    < np.array(og_logs["return_per_episode"]).mean()
                )
                or d["missing_actions"]
                or not d["mutual_exclusivity"]
            ):
                continue
            t_vals_candidates.append(t_vals[i].item())
        if len(t_vals_candidates) == 0:
            log.info("No thresholding candidate found!")
            return (
                SpecialStateCorridorFailureCode.FAIL_AT_THRESHOLD_NO_CANDIDATE
            )

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
                return (
                    SpecialStateCorridorFailureCode.FAIL_AT_THRESHOLD_NO_CANDIDATE
                )
            t_vals_candidates = threshold_json_dict["threshold_vals"]
        apply_threshold_with_candidate_list(t_vals_candidates)
    else:
        ret = threshold_model()
        threshold_json_dict = {}
        with open(model_dir / "threshold_val_candidates.json", "w") as f:
            if isinstance(ret, SpecialStateCorridorFailureCode):
                threshold_json_dict["threshold_success"] = False
                json.dump(threshold_json_dict, f)
                return ret
            t_vals_candidates = ret
            threshold_json_dict["threshold_success"] = True
            threshold_json_dict["threshold_vals"] = t_vals_candidates
            json.dump(threshold_json_dict, f)
        apply_threshold_with_candidate_list(t_vals_candidates)

    _simulate_with_print(_action_fn, f"{base_stage_name} (thresholded)")
    log.info(model.actor.conjunctions.weights.data)
    log.info(model.actor.disjunctions.weights.data)
    log.info("\n")

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
    obs, _ = single_env.reset()

    terminated = False
    truncated = False
    reward_sum = 0

    while not terminated and not truncated:
        input_encoding = [f"a_{obs['agent_location']}."]
        log.info(input_encoding)

        ctl = clingo.Control(["--warn=none"])
        show_statements = [
            f"#show disj_{i}/0." for i in range(model.action_size)
        ]
        ctl.add("base", [], " ".join(input_encoding + show_statements + rules))
        ctl.ground([("base", [])])
        with ctl.solve(yield_=True) as handle:  # type: ignore
            all_answer_sets = [str(a) for a in handle]

        if len(all_answer_sets) != 1:
            # No model or multiple answer sets, should not happen
            log.info(f"No model or multiple answer sets when evaluating rules.")
            return (
                SpecialStateCorridorFailureCode.FAIL_AT_RULE_EVAL_NOT_ONE_ANSWER_SET
            )

        if all_answer_sets[0] == "":
            log.info(f"No output action!")
            return (
                SpecialStateCorridorFailureCode.FAIL_AT_RULE_EVAL_MISSING_ACTION
            )

        output_classes = all_answer_sets[0].split(" ")
        if len(output_classes) == 0:
            log.info(f"No output action!")
            return (
                SpecialStateCorridorFailureCode.FAIL_AT_RULE_EVAL_MISSING_ACTION
            )
        output_classes_set = set([int(o[5:]) for o in output_classes])

        if len(output_classes_set) != 1:
            log.info(f"Output set: {output_classes_set} not exactly one item!")
            return (
                SpecialStateCorridorFailureCode.FAIL_AT_RULE_EVAL_MORE_THAN_ONE_ACTION
            )

        action = list(output_classes_set)[0]
        log.info(f"Action: {action}")
        obs, reward, terminated, truncated, _ = single_env.step(action)
        reward_sum += reward

    if truncated:
        log.info(f"Truncated: {reward_sum}")
        return SpecialStateCorridorFailureCode.FAIL_AT_RULE_EVAL_TRUNCATED

    log.info(f"Reward sum: {reward_sum}")
    return 0


def post_train_eval(eval_cfg: DictConfig):
    experiment_name = eval_cfg["experiment_name"]
    use_ndnf = "ndnf" in experiment_name
    assert use_ndnf

    use_state_no_as_obs = "sn" in experiment_name

    ret_dict: dict[int, list[int]] = dict(
        [(c.value, []) for c in SpecialStateCorridorFailureCode] + [(0, [])]
    )

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
        assert isinstance(model, SSCPPONDNFBasedAgent)
        model.to(DEVICE)
        model_state = torch.load(model_dir / "model.pth", map_location=DEVICE)
        model.load_state_dict(model_state)
        model.eval()

        log.info(f"Experiment {model_dir.name} loaded!")
        ret_code = post_training(
            model_dir=model_dir,
            model=model,
            envs=envs,
            single_env=single_env,
            process_obs=process_obs,
            env_max_steps=single_env.truncate_tolerance,
        )
        ret_dict[ret_code].append(s)  # use the random seed as the identifier
        log.info("======================================\n")

    for k, v in ret_dict.items():
        if k == 0:
            log.info(f"Success: {sorted(v)}")
        else:
            log.info(f"{SpecialStateCorridorFailureCode(k).name}: {sorted(v)}")


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
