# This script evaluates the ProbLog rules extracted from the NDNF-MT actor
# trained on the Blackjack environment.
import json
import logging
from pathlib import Path
import random
import sys
import traceback
from typing import Any

import hydra
import matplotlib.pyplot as plt
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


from blackjack_common import (
    construct_model,
    get_target_policy,
    decode_tuple_obs,
    BlackjackNDNFMutexTanhAgent,
    construct_single_environment,
    create_policy_plots_from_action_distribution,
)
from common import synthesize
from eval.blackjack_ppo_rl_eval_common import (
    eval_on_environments,
    ndnf_based_agent_cmp_target_csv,
)
from eval.blackjack_ppo_ndnf_mt_post_train_soft_extraction import (
    SECOND_PRUNE_MODEL_PTH_NAME,
)
from eval.problog_inference_common import (
    prolog_inference_gen_action_dist_for_all_states,
    prolog_inference_in_env_single_run,
)
from utils import post_to_discord_webhook


DEFAULT_GEN_SEED = 2
DEVICE = torch.device("cpu")
BASE_STORAGE_DIR = root / "blackjack_ppo_storage"
PROBLOG_EVAL_NUM_RUNS = 100
PROBLOG_EXAMPLE_GENERATION_NUM_RUNS = 10

logging.getLogger("problog").setLevel(logging.WARNING)
log = logging.getLogger()


def blackjack_problog_context_gen_fn(obs: tuple) -> list[str]:
    return [f"{b}::input({j})." for j, b in enumerate(decode_tuple_obs(obs))]


def problog_inference_generate_examples(
    problog_rules: list[str],
    num_runs: int = PROBLOG_EXAMPLE_GENERATION_NUM_RUNS,
    use_argmax: bool = False,
) -> list[dict[str, Any]]:
    env = construct_single_environment()
    return [
        prolog_inference_in_env_single_run(
            env=env,
            problog_rules=problog_rules,
            num_actions=2,
            context_problog_gen_fn=blackjack_problog_context_gen_fn,
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
                    "action": action,
                }
                for j, (obs, action, action_probs) in enumerate(e["trace"])
            ],
            "episode_reward": e["episode_reward"],
        }
    return json_dict


def problog_inference_on_envs(
    problog_rules: list[str],
    eval_num_runs: int = PROBLOG_EVAL_NUM_RUNS,
    use_argmax: bool = False,
) -> dict[str, Any]:
    env = construct_single_environment()
    logs: dict[str, Any] = {"return_per_episode": []}

    for _ in range(eval_num_runs):
        ret = prolog_inference_in_env_single_run(
            env=env,
            problog_rules=problog_rules,
            num_actions=2,
            context_problog_gen_fn=blackjack_problog_context_gen_fn,
            use_argmax=use_argmax,
        )
        logs["return_per_episode"].append(ret["episode_reward"])

    # Calculate win rate
    num_wins = np.sum(np.array(logs["return_per_episode"]) == 1)
    num_losses = np.sum(np.array(logs["return_per_episode"]) == -1)
    num_draws = np.sum(np.array(logs["return_per_episode"]) == 0)

    logs["num_wins"] = num_wins
    logs["num_losses"] = num_losses
    logs["num_draws"] = num_draws
    logs["win_rate"] = num_wins / eval_num_runs
    logs["avg_return_per_episode"] = np.mean(logs["return_per_episode"])

    return logs


def problog_inference_on_all_states(
    target_policy_csv_path: Path,
    problog_rules: list[str],
    agent: BlackjackNDNFMutexTanhAgent,
    pre_computed_problog_act_dist: np.ndarray | None = None,
) -> tuple[bool, np.ndarray]:
    target_policy = get_target_policy(target_policy_csv_path)
    all_states = list(target_policy.keys())

    ret = ndnf_based_agent_cmp_target_csv(target_policy_csv_path, agent, DEVICE)
    ndnf_mt_act_dist = ret["action_distribution"]

    if pre_computed_problog_act_dist is not None:
        log.info("Using pre-computed ProbLog action distribution...")
        problog_act_dist = pre_computed_problog_act_dist
    else:
        log.info("Computing ProbLog action distribution...")
        all_states_context_problog = [
            [f"{b}::input({j})." for j, b in enumerate(decode_tuple_obs(s))]
            for s in all_states
        ]
        problog_act_dist = prolog_inference_gen_action_dist_for_all_states(
            all_states_context_problog, problog_rules, 2
        )

    # Check if the distributions are the same to 3 decimal places
    close_dist = np.allclose(ndnf_mt_act_dist, problog_act_dist, atol=1e-3)

    return close_dist, problog_act_dist


def inference(
    problog_rules: list[str],
    target_policy_csv_path: Path,
    agent: BlackjackNDNFMutexTanhAgent,
    model_dir: Path,
    use_argmax: bool = False,
) -> dict[str, Any]:
    # Check for pre-computed ProbLog action distribution
    pre_computed_problog_act_dist = None
    if (model_dir / "problog_inference_act_dist.npy").exists():
        pre_computed_problog_act_dist = np.load(
            model_dir / "problog_inference_act_dist.npy"
        )

    close_dist, problog_act_dist = problog_inference_on_all_states(
        target_policy_csv_path,
        problog_rules,
        agent,
        pre_computed_problog_act_dist,
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
        eval_logs = eval_on_environments(agent, DEVICE, use_argmax=use_argmax)
        eval_logs.pop("num_frames_per_episode", None)
    else:
        log.info("Using ProbLog rules for evaluation...")
        eval_logs = problog_inference_on_envs(
            problog_rules, use_argmax=use_argmax
        )
    log.info(f"Win rate: {eval_logs['win_rate']}")
    log.info(f"Avg. return per episode: {eval_logs['avg_return_per_episode']}")

    # Generate examples
    examples = problog_inference_generate_examples(problog_rules)
    with open(model_dir / "problog_inference_examples.json", "w") as f:
        json.dump(parse_traces_to_json(examples), f, indent=4)

    # Generate policy plots
    plot = create_policy_plots_from_action_distribution(
        target_policy=get_target_policy(target_policy_csv_path),
        model_action_distribution=torch.tensor(problog_act_dist),
        model_name="Problog Rules",
        argmax=use_argmax,
        plot_diff=True,
    )
    plot.savefig(model_dir / "problog_policy_cmp_q.png")
    plt.close()

    return {
        "close_dist": close_dist,
        "problog_act_dist": problog_act_dist,
        "examples": examples,
        **eval_logs,
    }


def post_interpret_inference(eval_cfg: DictConfig):
    experiment_name = f"{eval_cfg['experiment_name']}"
    use_ndnf = "ndnf" in experiment_name

    assert use_ndnf and not eval_cfg["use_eo"] and eval_cfg["use_mt"]

    target_policy_csv_path = Path(eval_cfg["target_policy_csv_path"])
    if not target_policy_csv_path.exists():
        raise FileNotFoundError(
            f"The target policy csv file {target_policy_csv_path} does not exist!"
        )

    close_dist_list = []
    win_rate_list = []
    avg_return_per_episode_list = []

    for s in eval_cfg["multirun_seeds"]:
        # Load agent
        model_dir = BASE_STORAGE_DIR / f"{experiment_name}_{s}"
        model = construct_model(
            num_latent=eval_cfg["model_latent_size"],
            use_ndnf=use_ndnf,
            use_decode_obs=True,
            use_eo=False,
            use_mt=True,
            share_layer_with_critic=eval_cfg["share_layer_with_critic"],
        )
        assert isinstance(model, BlackjackNDNFMutexTanhAgent)
        assert (
            model_dir / SECOND_PRUNE_MODEL_PTH_NAME
        ).exists(), (
            "Please run the soft extraction first before inference with rules."
        )
        assert (
            model_dir / "problog_rules.pl"
        ).exists(), (
            "Please run the interpretation first before inference with rules."
        )
        model.to(DEVICE)
        model.eval()

        sd = torch.load(
            model_dir / SECOND_PRUNE_MODEL_PTH_NAME, map_location=DEVICE
        )
        model.load_state_dict(sd)

        with open(model_dir / "problog_rules.pl", "r") as f:
            problog_rules = f.readlines()
        problog_rules = [r.strip() for r in problog_rules]

        log.info(f"Interpretation of {model_dir.name}:")
        ret = inference(problog_rules, target_policy_csv_path, model, model_dir)
        if not ret["close_dist"]:
            log.info(
                f"Seed {s} has a different action distribution generated from"
                " Problog rules than the NDNF-MT agent."
            )
        else:
            close_dist_list.append(s)
        win_rate_list.append(ret["win_rate"])
        avg_return_per_episode_list.append(ret["avg_return_per_episode"])
        log.info("======================================")

    log.info(f"Close distributions runs: {close_dist_list}")
    log.info(
        f"Proportion: {len(close_dist_list) / len(eval_cfg['multirun_seeds'])}"
    )
    log.info(f"Avg. win rate: {np.mean(win_rate_list)}")
    log.info(f"Avg. return per episode: {np.mean(avg_return_per_episode_list)}")

    aggregated_logs = {}
    aggregated_logs["close_dist_list"] = close_dist_list
    for k, v in synthesize(win_rate_list, compute_ste=True).items():
        aggregated_logs[f"avg_win_rate_{k}"] = float(v)
    for k, v in synthesize(
        avg_return_per_episode_list, compute_ste=True
    ).items():
        aggregated_logs[f"avg_return_per_episode_{k}"] = float(v)

    with open("blackjack_problog_inference_aggregated_logs.json", "w") as f:
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
