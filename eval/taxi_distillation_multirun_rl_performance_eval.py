# This script evaluates the performance of the trained NDNF based models on the
# taxi environment and compare the actor to a target policy (either a Q table or
# a MLP model). The evaluation is done on the environments and all possible
# states the agent can encounter.
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

from neural_dnf.neural_dnf import (
    BaseNeuralDNF,
    NeuralDNF,
    NeuralDNFEO,
    NeuralDNFMutexTanh,
    NeuralDNFFullMutexTanh,
)

from eval.taxi_distillation_rl_eval_common import (
    eval_on_environments,
    eval_on_all_possible_states,
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

BASE_STORAGE_DIR = root / "taxi_distillation_storage"
DEFAULT_GEN_SEED = 2
DEVICE = torch.device("cpu")


log = logging.getLogger()


def single_ndnf_mt_eval(
    model: BaseNeuralDNF,
    device: torch.device = DEVICE,
    target_q_table: np.ndarray | None = None,
    target_action_dist: Categorical | None = None,
    use_argmax: bool = True,
) -> dict[str, Any]:
    env_eval_log = eval_on_environments(model, device, use_argmax=use_argmax)

    all_states_eval_log = eval_on_all_possible_states(
        ndnf_model=model,
        device=device,
        target_q_table=target_q_table,
        target_action_dist=target_action_dist,
    )

    final_log = {**env_eval_log, **all_states_eval_log}
    return final_log


def result_analysis(
    single_eval_results: list[OrderedDict[str, float]],
) -> dict[str, float]:
    np.set_printoptions(formatter={"float": lambda x: "{:.3f}".format(x)})
    aggregated_log: dict[str, float] = dict()

    num_models = len(single_eval_results)
    all_runs_return = np.array(
        [
            d[EnvEvalLogKeys.AVG_RETURN_PER_EPISODE.value]
            for d in single_eval_results
        ]
    )

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

    # Avg. KL of all models if present
    if StateEvalLogKeys.KL_DIV.value in single_eval_results[0]:
        all_runs_kl = np.array(
            [d[StateEvalLogKeys.KL_DIV.value] for d in single_eval_results]
        )
        avg_kl, std_kl, ste_kl = compute_mean_std_ste(all_runs_kl)
        log.info(f"Avg. KL of all runs: {avg_kl:.3f}")
        log.info(f"Std. KL of all runs: {std_kl:.3f}")
        log.info(f"Ste. KL of all runs: {ste_kl:.3f}")
        log.info("=====================================")

        aggregated_log["avg_kl"] = avg_kl
        aggregated_log["std_kl"] = std_kl
        aggregated_log["ste_kl"] = ste_kl

    # Avg. policy error rate of all models if present
    if (
        StateEvalLogKeys.POLICY_ERROR_RATE_CMP_TARGET.value
        in single_eval_results[0]
    ):
        all_runs_policy_error_rate = np.array(
            [
                d[StateEvalLogKeys.POLICY_ERROR_RATE_CMP_TARGET.value]
                for d in single_eval_results
            ]
        )
        avg_policy_error_rate, std_policy_error_rate, ste_policy_error_rate = (
            compute_mean_std_ste(all_runs_policy_error_rate)
        )
        log.info(
            f"Avg. policy error rate compared to target policy of all runs: {avg_policy_error_rate:.3f}"
        )
        log.info(
            f"Std. policy error rate compared to target policy of all runs: {std_policy_error_rate:.3f}"
        )
        log.info(
            f"Ste. policy error rate compared to target policy of all runs: {ste_policy_error_rate:.3f}"
        )
        log.info("=====================================")

        aggregated_log["avg_policy_error_rate_cmp_target"] = (
            avg_policy_error_rate
        )
        aggregated_log["std_policy_error_rate_cmp_target"] = (
            std_policy_error_rate
        )
        aggregated_log["ste_policy_error_rate_cmp_target"] = (
            ste_policy_error_rate
        )

    # Check for ME and MA
    if StateEvalLogKeys.ME.value in single_eval_results[0]:
        all_runs_me = np.array(
            [d.get(StateEvalLogKeys.ME.value) for d in single_eval_results]
        )
        all_runs_me_violations_count = np.array(
            [
                d.get(StateEvalLogKeys.ME_COUNT.value, 0)
                for d in single_eval_results
            ]
        )
        all_runs_ma = np.array(
            [d[StateEvalLogKeys.MA.value] for d in single_eval_results]
        )
        all_runs_ma_count = np.array(
            [
                d.get(StateEvalLogKeys.MA_COUNT.value, 0)
                for d in single_eval_results
            ]
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
    assert "ndnf_mt" in experiment_name

    use_decode_obs = eval_cfg["use_decode_obs"]
    model_type_str = eval_cfg["model_type"]
    use_argmax = eval_cfg.get("use_argmax", True)

    model_type: BaseNeuralDNF = {
        "plain": NeuralDNF,
        "eo": NeuralDNFEO,
        "mt": NeuralDNFMutexTanh,
        "fmt": NeuralDNFFullMutexTanh,
    }[model_type_str]

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

    single_eval_results = []

    for s in eval_cfg["multirun_seeds"]:
        # Load agent
        full_experiment_name = f"{experiment_name}_{s}"
        model_dir = BASE_STORAGE_DIR / full_experiment_name
        model = model_type(
            num_preds=(
                N_DECODE_OBSERVATION_SIZE
                if use_decode_obs
                else N_OBSERVATION_SIZE
            ),
            num_conjuncts=eval_cfg["num_conjunctions"],
            n_out=N_ACTIONS,
            delta=1.0,
        )

        assert isinstance(model, BaseNeuralDNF)

        model.to(DEVICE)
        model_state = torch.load(model_dir / "model.pth", map_location=DEVICE)
        model.load_state_dict(model_state)
        model.eval()

        log.info(f"Experiment {model_dir.name} loaded!")

        eval_log = single_ndnf_mt_eval(
            model=model,
            device=DEVICE,
            target_q_table=target_q_table,
            target_action_dist=target_action_dist,
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

    torch.autograd.set_detect_anomaly(True)  # type: ignore

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
