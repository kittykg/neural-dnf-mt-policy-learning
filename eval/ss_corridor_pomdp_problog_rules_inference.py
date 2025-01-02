# This script evaluates the ProbLog rules extracted from the NDNF-MT actor
# trained on the Blackjack environment.
import json
import logging
from pathlib import Path
import random
import sys
import traceback
from typing import Any

import gymnasium as gym
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

from common import synthesize
from eval.problog_inference_common import (
    prolog_inference_in_env_single_run,
    prolog_inference_gen_action_dist_for_all_states,
)
from eval.ss_corridor_ppo_pomdp_ndnf_multirun_eval import (
    simulate_fn,
    NUM_PROCESSES,
)
from eval.ss_corridor_ppo_pomdp_ndnf_mt_post_train_interpretation import (
    SECOND_PRUNE_MODEL_PTH_NAME,
)
from ss_corridor_ppo import (
    construct_model,
    construct_single_environment,
    make_env,
    SSCPPONDNFMutexTanhAgent,
    ss_corridor_preprocess_obs,
)
from utils import post_to_discord_webhook


ALL_POSSIBLE_WALL_STATUS = [
    [-1, -1],  # no wall on either side
    [1, -1],  # wall on the left
    [-1, 1],  # wall on the right
]

BASE_STORAGE_DIR = root / "ssc_ppo_storage"
DEFAULT_GEN_SEED = 2
DEVICE = torch.device("cpu")
NDNF_MT_EVAL_NUM_EPISODES = 1000000
PROBLOG_EVAL_NUM_RUNS = 100
PROBLOG_EXAMPLE_GENERATION_NUM_RUNS = 10

logging.getLogger("problog").setLevel(logging.WARNING)
log = logging.getLogger()


def ss_corridor_problog_context_gen_fn(obs: dict[str, np.ndarray]) -> list[str]:
    return [f"{b}::input({j})." for j, b in enumerate(obs["wall_status"])]


def problog_inference_generate_examples(
    problog_rules: list[str],
    eval_cfg: DictConfig,
    num_runs: int = PROBLOG_EXAMPLE_GENERATION_NUM_RUNS,
    use_argmax: bool = False,
) -> list[dict[str, Any]]:
    env = construct_single_environment(eval_cfg)
    return [
        prolog_inference_in_env_single_run(
            env=env,
            problog_rules=problog_rules,
            num_actions=2,
            context_problog_gen_fn=ss_corridor_problog_context_gen_fn,
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
                    "obs": tuple(obs),
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
    eval_cfg: DictConfig,
    eval_num_runs: int = PROBLOG_EVAL_NUM_RUNS,
    use_argmax: bool = False,
) -> dict[str, Any]:
    env = construct_single_environment(eval_cfg)
    logs: dict[str, Any] = {
        "return_per_episode": [],
        "num_steps_per_episode": [],
    }

    for _ in range(eval_num_runs):
        ret = prolog_inference_in_env_single_run(
            env=env,
            problog_rules=problog_rules,
            num_actions=2,
            context_problog_gen_fn=ss_corridor_problog_context_gen_fn,
            use_argmax=use_argmax,
        )
        logs["return_per_episode"].append(ret["episode_reward"])
        logs["num_steps_per_episode"].append(ret["num_frames"])

    logs["avg_return_per_episode"] = np.mean(logs["return_per_episode"])
    logs["avg_num_steps_per_episode"] = np.mean(logs["num_steps_per_episode"])

    return logs


def problog_inference_on_all_states(
    problog_rules: list[str],
    agent: SSCPPONDNFMutexTanhAgent,
    pre_computed_problog_act_dist: np.ndarray | None = None,
) -> tuple[bool, np.ndarray]:
    with torch.no_grad():
        ndnf_mt_act_dist = (
            agent.get_action_distribution(
                {
                    "input": torch.Tensor(
                        ALL_POSSIBLE_WALL_STATUS, device=DEVICE
                    ).float()
                }
            )
            .probs.cpu()  # type: ignore
            .numpy()
        )

    if pre_computed_problog_act_dist is not None:
        log.info("Using pre-computed ProbLog action distribution...")
        problog_act_dist = pre_computed_problog_act_dist
    else:
        log.info("Computing ProbLog action distribution...")
        all_states_context_problog = [
            [f"{(b + 1) / 2}::input({j})." for j, b in enumerate(s)]
            for s in ALL_POSSIBLE_WALL_STATUS
        ]
        problog_act_dist = prolog_inference_gen_action_dist_for_all_states(
            all_states_context_problog, problog_rules, 2
        )

    # Check if the distributions are the same to 3 decimal places
    close_dist = np.allclose(ndnf_mt_act_dist, problog_act_dist, atol=1e-3)

    return close_dist, problog_act_dist


def ndnf_mt_agent_inference_on_envs(
    agent: SSCPPONDNFMutexTanhAgent, eval_cfg: DictConfig
) -> dict[str, Any]:
    envs = gym.vector.SyncVectorEnv(
        [make_env(eval_cfg, i, i, False) for i in range(NUM_PROCESSES)]
    )
    corridor_length = envs.single_observation_space["agent_location"].n  # type: ignore
    logs = simulate_fn(
        envs=envs,
        model=agent,
        process_obs=lambda obs: ss_corridor_preprocess_obs(
            use_state_no_as_obs=False,
            use_ndnf=True,
            corridor_length=corridor_length,
            obs=obs,
            device=DEVICE,
        ),
        num_episodes=NDNF_MT_EVAL_NUM_EPISODES,
    )
    envs.close()

    # Remove "action_distribution" in the logs
    logs.pop("action_distribution", None)

    # Rename "num_frames_per_episode" to "num_steps_per_episode"
    logs["num_steps_per_episode"] = logs.pop("num_frames_per_episode")

    # Add average return and number of steps per episode
    logs["avg_return_per_episode"] = np.mean(logs["return_per_episode"])
    logs["avg_num_steps_per_episode"] = np.mean(logs["num_steps_per_episode"])

    return logs


def inference(
    problog_rules: list[str],
    agent: SSCPPONDNFMutexTanhAgent,
    eval_cfg: DictConfig,
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
        eval_logs = ndnf_mt_agent_inference_on_envs(agent, eval_cfg)
    else:
        log.info("Using ProbLog rules for evaluation...")
        eval_logs = problog_inference_on_envs(
            problog_rules, eval_cfg, use_argmax
        )
    log.info(f"Avg. return per episode: {eval_logs['avg_return_per_episode']}")
    log.info(
        f"Avg. num steps per episode: {eval_logs['avg_num_steps_per_episode']}"
    )

    # Generate examples
    examples = problog_inference_generate_examples(
        problog_rules, eval_cfg, use_argmax=use_argmax
    )
    with open(model_dir / "problog_inference_examples.json", "w") as f:
        json.dump(parse_traces_to_json(examples), f, indent=4)

    return {
        "close_dist": close_dist,
        "problog_act_dist": problog_act_dist,
        "examples": examples,
        **eval_logs,
    }


def post_interpret_inference(eval_cfg: DictConfig):
    experiment_name = f"{eval_cfg['experiment_name']}"
    use_ndnf = "ndnf" in experiment_name

    assert (
        use_ndnf and "mt" in experiment_name
    ), "This evaluation script is only for the NDNF-MT agent."

    close_dist_list = []
    avg_return_list = []
    avg_num_steps_list = []

    for s in eval_cfg["multirun_seeds"]:
        # Load agent
        model_dir = BASE_STORAGE_DIR / f"{experiment_name}_{s}"
        model = construct_model(
            num_inputs=2,
            num_latent=eval_cfg["model_latent_size"],
            action_size=2,
            use_ndnf=use_ndnf,
            use_eo="eo" in experiment_name,
            use_mt="mt" in experiment_name,
        )
        assert isinstance(model, SSCPPONDNFMutexTanhAgent)
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
        ret = inference(problog_rules, model, eval_cfg, model_dir)
        if not ret["close_dist"]:
            log.info(
                f"Seed {s} has a different action distribution generated from"
                " Problog rules than the NDNF-MT agent."
            )
        else:
            close_dist_list.append(s)
        avg_return_list.append(ret["avg_return_per_episode"])
        avg_num_steps_list.append(ret["avg_num_steps_per_episode"])
        log.info("======================================")

    log.info(f"Close distributions runs: {close_dist_list}")
    log.info(
        f"Proportion: {len(close_dist_list) / len(eval_cfg['multirun_seeds'])}"
    )

    # Flatten the return lists
    synthesized_logs = synthesize(avg_return_list, compute_ste=True)
    log.info(f"Avg. return per episode: {synthesized_logs['mean']}")
    log.info(f"STE: {synthesized_logs['ste']}")

    # Save the synthesized logs
    with open(
        f"{experiment_name}_problog_inference_aggregated_log.json",
        "w",
    ) as f:
        json.dump(
            {k: float(v) for k, v in synthesized_logs.items()}, f, indent=4
        )


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

    if mp.get_start_method() != "forkserver":
        mp.set_start_method("forkserver", force=True)

    run_eval()
