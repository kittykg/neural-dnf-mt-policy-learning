# This script soft extracts (prune and threshold on conjunctions) the NDNF-MT
# model distilled from an MLP actor in Taxi environment, based on the comparison
# result on that MLP actor. This script is the pre-requisite for the problog
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
import numpy.typing as npt
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.distributions.categorical import Categorical


file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from neural_dnf.neural_dnf import NeuralDNFMutexTanh
from neural_dnf.post_training import prune_neural_dnf

from eval.taxi_distillation_rl_eval_common import (
    eval_on_all_possible_states,
    eval_on_environments,
    EnvEvalLogKeys,
    StateEvalLogKeys,
)
from taxi_common import N_ACTIONS, N_OBSERVATION_SIZE, N_DECODE_OBSERVATION_SIZE
from taxi_distillation import (
    load_mlp_model,
    generate_data_from_mlp,
    load_target_q_table,
)
from utils import post_to_discord_webhook


DEFAULT_GEN_SEED = 2
DEVICE = torch.device("cpu")
BASE_STORAGE_DIR = root / "taxi_distillation_storage"

FIRST_PRUNE_MODEL_PTH_NAME = "model_soft_mr_pruned.pth"
THRESHOLD_MODEL_PTH_NAME = "soft_thresholded_model.pth"
THRESHOLD_JSON_NAME = "soft_threshold_val_candidates.json"
SECOND_PRUNE_MODEL_PTH_NAME = "model_soft_2nd_mr_pruned.pth"

log = logging.getLogger()


def post_training(
    model: NeuralDNFMutexTanh,
    model_dir: Path,
    target_q_table: np.ndarray | None = None,
    target_action_dist: Categorical | None = None,
) -> dict[str, Any]:
    # Helper functions
    def _eval() -> dict[str, Any]:
        states_eval_logs = eval_on_all_possible_states(
            ndnf_model=model,
            device=DEVICE,
            target_q_table=target_q_table,
            target_action_dist=target_action_dist,
        )
        env_eval_logs = eval_on_environments(
            ndnf_model=model,
            device=DEVICE,
        )
        return {**states_eval_logs, **env_eval_logs}

    def _simulate_with_print(model_name: str) -> dict[str, Any]:
        logs = _eval()
        log.info(f"Model: {model_name}")
        log.info(
            "Has truncation in environment: "
            f"{logs[EnvEvalLogKeys.HAS_TRUNC.value]}"
        )
        log.info(
            "No. ME violations (all states): "
            f"{logs.get(StateEvalLogKeys.ME_COUNT.value, 0)}"
        )
        log.info(
            "No. missing actions (all states): "
            f"{logs.get(StateEvalLogKeys.MA_COUNT.value, 0)}"
        )
        log.info(
            "Policy error compared to target: "
            f"{logs[StateEvalLogKeys.POLICY_ERROR_RATE_CMP_TARGET.value]}"
        )
        return logs

    # Stage 1: Evaluate the model post-training only on normal states
    _simulate_with_print("NDNF MT Soft")
    log.info("======================================")

    # Stage 2: Prune the model
    def prune_model() -> None:
        log.info("Pruning the model...")

        def pruning_cmp_fn(og_log, new_log):
            # Need to guarantee that the environment is not truncated
            env_trunc = new_log[EnvEvalLogKeys.HAS_TRUNC.value]
            if env_trunc:
                return False

            # Check if the action distribution is close
            # We don't use kl divergence just yet
            og_action_dist = og_log[StateEvalLogKeys.ACTION_DISTRIBUTION.value]
            new_action_dist = new_log[
                StateEvalLogKeys.ACTION_DISTRIBUTION.value
            ]

            return np.allclose(og_action_dist, new_action_dist, atol=1e-2)

        sd_list = []
        prune_count = 0

        while True:
            log.info(f"Pruning iteration: {prune_count + 1}")
            prune_result_dict = prune_neural_dnf(
                model,
                _eval,
                {},
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
    og_conj_weight = model.conjunctions.weights.data.clone()

    # calculate the kl divergence between two distributions p and q, where p is
    # the true distribution (action dist after pruning) and q is the estimated
    # distribution (thresholded action dist)
    def kl_divergence(p: npt.NDArray, q: npt.NDArray) -> float:
        return float(np.mean(np.where(p != 0, p * np.log(p / q), 0)))

    # TODO: Check this
    # Convert the agent to plain NDNF agent
    # model = model.to_ndnf_agent()  # type: ignore
    model.eval()

    def threshold_model() -> dict[str, float]:
        log.info("Thresholding the model conjunction...")

        conj_min = torch.min(model.conjunctions.weights.data)
        conj_max = torch.max(model.conjunctions.weights.data)
        threshold_upper_bound = round(
            (torch.Tensor([conj_min, conj_max]).abs().max() + 0.01).item(),
            2,
        )
        log.info(f"Threshold upper bound: {threshold_upper_bound}")

        t_vals = torch.arange(0, threshold_upper_bound, 0.01)
        result_dicts = []

        for v in t_vals:
            model.conjunctions.weights.data = (
                (torch.abs(og_conj_weight) > v)
                * torch.sign(og_conj_weight)
                * 6.0
            )
            r = eval_on_all_possible_states(
                ndnf_model=model,
                device=DEVICE,
                target_q_table=target_q_table,
                target_action_dist=target_action_dist,
            )
            r["t_val"] = v.item()
            r["kl"] = kl_divergence(
                post_prune_logs[StateEvalLogKeys.ACTION_DISTRIBUTION.value],
                r[StateEvalLogKeys.ACTION_DISTRIBUTION.value],
            )
            result_dicts.append(r)

        log.info("Proceed to threshold based on KL...")
        filtered_result_dicts = [
            d for d in result_dicts if not d[EnvEvalLogKeys.HAS_TRUNC.value]
        ]
        second_sorted_result_dict = sorted(
            filtered_result_dicts,
            key=lambda d: (
                d["kl"],
                d[StateEvalLogKeys.POLICY_ERROR_RATE_CMP_TARGET.value],
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
        model.conjunctions.weights.data = (
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
    kl = kl_divergence(
        post_prune_logs[StateEvalLogKeys.ACTION_DISTRIBUTION.value],
        second_prune_logs[StateEvalLogKeys.ACTION_DISTRIBUTION.value],
    )
    log.info(f"KL divergence cmp to after 1st prune: {kl}")

    return second_prune_logs


def post_train_eval(eval_cfg: DictConfig) -> dict[str, Any]:
    experiment_name = f"{eval_cfg['experiment_name']}"
    assert eval_cfg["model_type"] == "mt"
    use_decode_obs = eval_cfg["use_decode_obs"]

    target_q_table = None
    target_action_dist = None

    if eval_cfg["distillation_mlp"]["mlp_model_path"] is not None:
        # Pass a dummy config to load_model
        distillation_mlp_cfg: dict[str, Any] = OmegaConf.to_container(
            eval_cfg["distillation_mlp"].copy()
        )  # type: ignore
        mlp_model_path_str = distillation_mlp_cfg.pop("mlp_model_path")
        mlp_model = load_mlp_model(
            model_architecture_cfg=distillation_mlp_cfg,
            mlp_model_path_str=mlp_model_path_str,
            device=DEVICE,
        )
        _, target_action_dist = generate_data_from_mlp(mlp_model, DEVICE)

    else:
        assert (
            eval_cfg["distillation_tab_q"]["tab_q_path"] is not None
        ), "Either mlp_model_path or tab_q_path must be provided"

        tab_q_path_str = eval_cfg["distillation_tab_q"]["tab_q_path"]
        target_q_table = load_target_q_table(tab_q_path_str)

    policy_error_cmp_to_target_list: list[float] = []

    log.info(f"Start soft extraction on {experiment_name}")
    log.info("======================================")

    for s in eval_cfg["multirun_seeds"]:
        # Load agent
        model_dir = BASE_STORAGE_DIR / f"{experiment_name}_{s}"
        model = NeuralDNFMutexTanh(
            num_preds=(
                N_DECODE_OBSERVATION_SIZE
                if use_decode_obs
                else N_OBSERVATION_SIZE
            ),
            num_conjuncts=eval_cfg["num_conjunctions"],
            n_out=N_ACTIONS,
            delta=1.0,
        )
        model.to(DEVICE)
        model_state = torch.load(model_dir / "model.pth", map_location=DEVICE)
        model.load_state_dict(model_state)
        model.eval()

        log.info(f"Experiment {model_dir.name} loaded!")
        ret = post_training(
            model, model_dir, target_q_table, target_action_dist
        )
        policy_error_cmp_to_target_list.append(
            ret[StateEvalLogKeys.POLICY_ERROR_RATE_CMP_TARGET.value]
        )
        log.info("======================================")
        log.info("======================================")

    log.info(
        f"Average policy error compared to target: {np.array(policy_error_cmp_to_target_list).mean()}"
    )

    return {
        "avg_policy_error_cmp_to_target": np.array(
            policy_error_cmp_to_target_list
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
            msg_body += f"Average policy error compared to target: {ret_dict['avg_policy_error_cmp_to_target']}"
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
    run_eval()
