# This script evaluates the ProbLog rules extracted from the NDNF-MT actor
# trained on the Taxi environment.
# Since ProbLog inference is computationally heavy, we do not run ProbLog at all
# for this evaluation specifically. Instead, we check that only one rule can be
# fired per state. If this is satisfied, then we use the DNF-MT actor for
# evaluation.
# We do leave all the ProbLog inference code in this script, but it is not
# executed.
from enum import IntEnum
import json
import logging
from pathlib import Path
import random
import sys
import traceback
from typing import Any

import clingo
import hydra
import numpy as np
from omegaconf import DictConfig
import torch
from torch.distributions import Categorical


file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass


from neural_dnf import NeuralDNFMutexTanh

from common import synthesize
from eval.taxi_distillation_ndnf_mt_post_train_soft_extraction import (
    SECOND_PRUNE_MODEL_PTH_NAME,
)
from eval.taxi_distillation_rl_eval_common import (
    eval_on_all_possible_states,
    eval_on_environments,
    get_target_q_table_and_action_dist,
    EnvEvalLogKeys,
    StateEvalLogKeys,
)
from eval.problog_inference_common import (
    prolog_inference_gen_action_dist_for_all_states,
    prolog_inference_in_env_single_run,
)

from taxi_common import (
    N_OBSERVATION_SIZE,
    N_DECODE_OBSERVATION_SIZE,
    N_ACTIONS,
    construct_single_environment,
    split_all_states_to_reachable_and_non,
    decode,
    convert_n_to_one_hot,
)
from utils import post_to_discord_webhook


BASE_STORAGE_DIR = root / "taxi_distillation_storage"
DEFAULT_GEN_SEED = 2
DEVICE = torch.device("cpu")
PROBLOG_EVAL_NUM_RUNS = 100
PROBLOG_EXAMPLE_GENERATION_NUM_RUNS = 10
TAXI_ENV_POSSIBLE_STATES, _ = split_all_states_to_reachable_and_non()


logging.getLogger("problog").setLevel(logging.WARNING)
log = logging.getLogger()


class TaxiInferenceVerificationFailureCode(IntEnum):
    NO_STABLE_MODEL = -1
    MULTIPLE_STABLE_MODELS = -2
    NO_OUTPUT_RULE = -3
    MULTIPLE_OUTPUT_RULES = -4


# ============================================================================ #
#                              ProbLog Inference                               #
# ============================================================================ #


def taxi_problog_context_gen_fn(use_decode_obs: bool, obs: int) -> list[str]:
    if use_decode_obs:
        input_array = decode(obs)
    else:
        input_array = convert_n_to_one_hot(obs)
    return [f"{b}::input({j})." for j, b in enumerate(input_array)]


def problog_inference_generate_examples(
    problog_rules: list[str],
    use_decode_obs: bool,
    num_runs: int = PROBLOG_EXAMPLE_GENERATION_NUM_RUNS,
    use_argmax: bool = False,
) -> list[dict[str, Any]]:
    env = construct_single_environment()
    return [
        prolog_inference_in_env_single_run(
            env=env,
            problog_rules=problog_rules,
            num_actions=N_ACTIONS,
            context_problog_gen_fn=lambda obs: taxi_problog_context_gen_fn(
                use_decode_obs, obs
            ),
            use_argmax=use_argmax,
        )
        for _ in range(num_runs)
    ]


def parse_traces_to_json(examples: list[dict[str, Any]]) -> dict[str, Any]:
    json_dict = {}
    for i, e in enumerate(examples):
        json_dict[i] = {
            "trace": [
                {
                    "time_step": j,
                    "obs": obs,
                    "action_0_prob": action_probs[0],
                    "action_1_prob": action_probs[1],
                    "action_2_prob": action_probs[2],
                    "action_3_prob": action_probs[3],
                    "action_4_prob": action_probs[4],
                    "action_5_prob": action_probs[5],
                    "action": action,
                }
                for j, (obs, action, action_probs) in enumerate(e["trace"])
            ],
            "episode_reward": e["episode_reward"],
        }
    return json_dict


def problog_inference_on_envs(
    problog_rules: list[str],
    use_decode_obs: bool,
    eval_num_runs: int = PROBLOG_EVAL_NUM_RUNS,
    use_argmax: bool = False,
) -> dict[str, Any]:
    env = construct_single_environment()
    logs: dict[str, Any] = {"return_per_episode": []}

    for _ in range(eval_num_runs):
        ret = prolog_inference_in_env_single_run(
            env=env,
            problog_rules=problog_rules,
            num_actions=N_ACTIONS,
            context_problog_gen_fn=lambda obs: taxi_problog_context_gen_fn(
                use_decode_obs, obs
            ),
            use_argmax=use_argmax,
        )
        logs["return_per_episode"].append(ret["episode_reward"])

    logs["avg_return_per_episode"] = np.mean(logs["return_per_episode"])

    return logs


def problog_inference_on_all_states(
    problog_rules: list[str],
    model: NeuralDNFMutexTanh,
    use_decode_obs: bool,
    pre_computed_problog_act_dist: np.ndarray | None = None,
    target_q_table: np.ndarray | None = None,
    target_action_dist: Categorical | None = None,
) -> tuple[bool, np.ndarray]:
    states_eval_logs = eval_on_all_possible_states(
        ndnf_model=model,
        device=DEVICE,
        target_q_table=target_q_table,
        target_action_dist=target_action_dist,
    )
    ndnf_mt_act_dist = states_eval_logs[
        StateEvalLogKeys.ACTION_DISTRIBUTION.value
    ]

    if pre_computed_problog_act_dist is not None:
        log.info("Using pre-computed ProbLog action distribution...")
        problog_act_dist = pre_computed_problog_act_dist
    else:
        log.info("Computing ProbLog action distribution...")
        all_states_context_problog = [
            taxi_problog_context_gen_fn(use_decode_obs, s)
            for s in TAXI_ENV_POSSIBLE_STATES
        ]
        problog_act_dist = prolog_inference_gen_action_dist_for_all_states(
            all_states_context_problog, problog_rules, N_ACTIONS
        )

    # Check if the distributions are the same to 3 decimal places
    close_dist = np.allclose(ndnf_mt_act_dist, problog_act_dist, atol=1e-3)

    return close_dist, problog_act_dist


def inference_with_problog(
    problog_rules: list[str],
    model: NeuralDNFMutexTanh,
    model_dir: Path,
    use_decode_obs: bool,
    use_argmax: bool = False,
    target_q_table: np.ndarray | None = None,
    target_action_dist: Categorical | None = None,
) -> dict[str, Any]:
    # Check for pre-computed ProbLog action distribution
    pre_computed_problog_act_dist = None
    if (model_dir / "problog_inference_act_dist.npy").exists():
        pre_computed_problog_act_dist = np.load(
            model_dir / "problog_inference_act_dist.npy"
        )

    close_dist, problog_act_dist = problog_inference_on_all_states(
        problog_rules=problog_rules,
        model=model,
        use_decode_obs=use_decode_obs,
        pre_computed_problog_act_dist=pre_computed_problog_act_dist,
        target_q_table=target_q_table,
        target_action_dist=target_action_dist,
    )
    if pre_computed_problog_act_dist is None:
        np.save(model_dir / "problog_inference_act_dist.npy", problog_act_dist)

    # ProbLog inference is computationally heaving and takes about 3s per
    # simulation. If the NDNF-MT agent's action distribution is close to the
    # ProbLog distribution, we can use the NDNF-MT instead of the ProbLog rules
    log.info(
        f"Problog action distribution close to neural DNF-MT agent: {close_dist}"
    )
    if close_dist:
        log.info("Using NDNF-MT agent for evaluation...")
        eval_logs = eval_on_environments(model, DEVICE, use_argmax=use_argmax)
        eval_logs.pop("num_frames_per_episode", None)
    else:
        log.info("Using ProbLog rules for evaluation...")
        eval_logs = problog_inference_on_envs(
            problog_rules, use_decode_obs, use_argmax=use_argmax
        )
    log.info(f"Avg. return per episode: {eval_logs['avg_return_per_episode']}")

    # Generate examples
    examples = problog_inference_generate_examples(
        problog_rules, use_decode_obs, use_argmax=use_argmax
    )
    with open(model_dir / "problog_inference_examples.json", "w") as f:
        json.dump(parse_traces_to_json(examples), f, indent=4)

    return {
        "close_dist": close_dist,
        "problog_act_dist": problog_act_dist,
        "examples": examples,
        **eval_logs,
    }


# ============================================================================ #
#                   NDNF-MT Inference with ASP verification                    #
# ============================================================================ #


def check_asp_rules_mutual_exclusive(
    asp_rules_for_problog: list[str],
) -> tuple[bool, TaxiInferenceVerificationFailureCode | None]:
    """
    This function checks the ASP rules used before ProbLog inference are
    mutually exclusive. That is, for a given state, only one rule should be
    fired, and this is enforced over all possible states.
    """
    # Verify that the ASP rules only have 1 fired per state
    show_statements = [f"#show rule/1."]

    for obs in TAXI_ENV_POSSIBLE_STATES:
        context_program = [f"input({obs})."]
        ctl = clingo.Control(["--warn=none"])
        ctl.add(
            "base",
            [],
            " ".join(context_program + asp_rules_for_problog + show_statements),
        )
        ctl.ground([("base", [])])
        with ctl.solve(yield_=True) as handle:  # type: ignore
            all_answer_sets = [str(a) for a in handle]

            if len(all_answer_sets) == 0:
                # No model or multiple answer sets, should not happen
                log.info(f"No model when evaluating rules.")
                return (
                    False,
                    TaxiInferenceVerificationFailureCode.NO_STABLE_MODEL,
                )
            elif len(all_answer_sets) > 1:
                # Multiple models, should not happen
                log.info(f"Multiple models when evaluating rules.")
                return (
                    False,
                    TaxiInferenceVerificationFailureCode.MULTIPLE_STABLE_MODELS,
                )

            answer_set = all_answer_sets[0]

            if answer_set == "":
                log.info(f"No output rule!")
                return (
                    False,
                    TaxiInferenceVerificationFailureCode.NO_OUTPUT_RULE,
                )

            output_rules = answer_set.split(" ")
            if len(output_rules) > 1:
                log.info(f"Output classes length is not 1 at obs {obs}")
                return (
                    False,
                    TaxiInferenceVerificationFailureCode.MULTIPLE_OUTPUT_RULES,
                )

    return True, None


def inference_with_asp_verification(
    asp_rules_for_problog: list[str],
    model: NeuralDNFMutexTanh,
    use_argmax: bool = False,
    target_q_table: np.ndarray | None = None,
    target_action_dist: Categorical | None = None,
):

    log.info(
        "Checking if ASP rules for ProbLog satisfy mutual exclusion every "
        "state..."
    )
    passed, error_code = check_asp_rules_mutual_exclusive(asp_rules_for_problog)
    if not passed:
        log.info(f"Verification failed with error code {error_code}")
        return {
            "verification_passed": False,
            "verification_error_code": error_code,
        }

    log.info("Verification successful. Using NDNF-MT agent for evaluation...")
    eval_env_logs = eval_on_environments(model, DEVICE, use_argmax=use_argmax)
    eval_state_logs = eval_on_all_possible_states(
        model, DEVICE, target_q_table, target_action_dist
    )
    log.info(
        f"Avg. return per episode: {eval_env_logs[EnvEvalLogKeys.AVG_RETURN_PER_EPISODE.value]}"
    )
    log.info(f"Has truncation: {eval_env_logs[EnvEvalLogKeys.HAS_TRUNC.value]}")

    return {
        "verification_passed": True,
        "avg_return_per_episode": eval_env_logs[
            EnvEvalLogKeys.AVG_RETURN_PER_EPISODE.value
        ],
        "has_truncation": eval_env_logs[EnvEvalLogKeys.HAS_TRUNC.value],
        "kl_div": eval_state_logs[StateEvalLogKeys.KL_DIV.value],
        "policy_error_rate_cmp_target": eval_state_logs[
            StateEvalLogKeys.POLICY_ERROR_RATE_CMP_TARGET.value
        ],
    }


def post_interpret_inference(eval_cfg: DictConfig):
    experiment_name = f"{eval_cfg['experiment_name']}"
    assert eval_cfg["model_type"] == "mt"
    use_decode_obs = eval_cfg["use_decode_obs"]

    target_q_table, target_action_dist = get_target_q_table_and_action_dist(
        eval_cfg, DEVICE
    )

    verification_pass_list = []

    target_metric_str = [
        "avg_return_per_episode",
        "has_truncation",
        "kl_div",
        "policy_error_rate_cmp_target",
    ]
    target_dict = {k: [] for k in target_metric_str}

    for s in eval_cfg["multirun_seeds"]:
        # Load agent
        model_dir = BASE_STORAGE_DIR / f"{experiment_name}_{s}"
        model = NeuralDNFMutexTanh(
            n_in=(
                N_DECODE_OBSERVATION_SIZE
                if use_decode_obs
                else N_OBSERVATION_SIZE
            ),
            n_conjunctions=eval_cfg["num_conjunctions"],
            n_out=N_ACTIONS,
            delta=1.0,
        )

        assert (
            model_dir / SECOND_PRUNE_MODEL_PTH_NAME
        ).exists(), (
            "Please run the soft extraction first before inference with rules."
        )
        assert (
            model_dir / "asp_rules_for_problog.pl"
        ).exists(), (
            "Please run the interpretation first before inference with rules."
        )
        model.to(DEVICE)
        model.eval()

        sd = torch.load(
            model_dir / SECOND_PRUNE_MODEL_PTH_NAME, map_location=DEVICE
        )
        model.load_state_dict(sd)

        with open(model_dir / "asp_rules_for_problog.pl", "r") as f:
            asp_rules_for_problog = f.readlines()
        asp_rules_for_problog = [r.strip() for r in asp_rules_for_problog]

        log.info(f"Interpretation of {model_dir.name}:")
        ret = inference_with_asp_verification(
            asp_rules_for_problog=asp_rules_for_problog,
            model=model,
            use_argmax=eval_cfg.get("use_argmax", False),
            target_q_table=target_q_table,
            target_action_dist=target_action_dist,
        )
        if not ret["verification_passed"]:
            continue

        verification_pass_list.append(s)
        for k in target_metric_str:
            target_dict[k].append(ret[k])
        log.info("======================================")

    log.info(f"Verified runs: {verification_pass_list}")
    log.info(
        f"Proportion: {len(verification_pass_list) / len(eval_cfg['multirun_seeds'])}"
    )
    log.info(
        f"Avg. return per episode: {np.mean(target_dict['avg_return_per_episode'])}"
    )

    aggregated_logs = {}
    aggregated_logs["verification_pass_list"] = verification_pass_list
    for k, l in target_dict.items():
        if k == "has_truncation":
            aggregated_logs[k] = any(l)
            continue

        for m, v in synthesize(l, compute_ste=True).items():
            aggregated_logs[f"{m}_{k}"] = float(v)

    with open("taxi_problog_inference_aggregated_logs.json", "w") as f:
        json.dump(aggregated_logs, f, indent=4)


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
        post_interpret_inference(eval_cfg)
        if use_discord_webhook:
            msg_body = f"Success!"
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
