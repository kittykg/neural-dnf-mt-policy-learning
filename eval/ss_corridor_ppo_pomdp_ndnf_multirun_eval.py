# This script evaluates the NDNF based agent on POMDP SpecialStateCorridor envs
# i.e. using wall status as observation of agent
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


from neural_dnf.post_training import prune_neural_dnf
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
from utils import post_to_discord_webhook


DEFAULT_GEN_SEED = 2
NUM_PROCESSES = 8
NUM_EPISODES = 100
DEVICE = torch.device("cpu")
BASE_STORAGE_DIR = root / "ssc_ppo_storage"

log = logging.getLogger()


def simulate_fn(
    envs: gym.vector.SyncVectorEnv,
    model: SSCPPONDNFBasedAgent,
    process_obs=Callable[[dict], Tensor],
    num_episodes=NUM_EPISODES,
) -> dict[str, Any]:
    logs: dict[str, Any] = {
        "num_frames_per_episode": [],
        "return_per_episode": [],
    }
    next_obs_dict, _ = envs.reset()
    next_obs = process_obs(next_obs_dict)
    next_obs_dict = {"input": next_obs}

    log_done_counter = 0
    log_episode_return = torch.zeros(NUM_PROCESSES, device=DEVICE)
    log_episode_num_frames = torch.zeros(NUM_PROCESSES, device=DEVICE)

    while log_done_counter < num_episodes:
        with torch.no_grad():
            actions = model.get_actions(
                preprocessed_obs=next_obs_dict, use_argmax=False
            )
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

    all_possible_wall_status = torch.Tensor(
        [
            [-1, -1],  # no wall on either side
            [1, -1],  # wall on the left
            [-1, 1],  # wall on the right
        ],
        device=DEVICE,
    ).float()

    with torch.no_grad():
        action_distribution = model.get_action_distribution(
            {"input": all_possible_wall_status}
        )
    logs["action_distribution"] = action_distribution.probs

    return logs


def post_training(
    model_dir: Path,
    model: SSCPPONDNFBasedAgent,
    envs: gym.vector.SyncVectorEnv,
    process_obs: Callable[[dict], Tensor],
) -> None:
    simulate = lambda: simulate_fn(envs, model, process_obs, NUM_EPISODES)

    def _simulate_with_print(model_name: str) -> dict[str, Any]:
        logs = simulate()

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
        log.info(f"Action distribution: {logs['action_distribution']}")
        return logs

    if isinstance(model, SSCPPONDNFEOAgent):
        base_stage_name = "NDNF EO"
        plain_ndnf_model = model.to_ndnf_agent()
    elif isinstance(model, SSCPPONDNFMutexTanhAgent):
        base_stage_name = "NDNF MT"
    else:
        base_stage_name = "NDNF"

    # Stage 1: Evaluate the model post-training
    _simulate_with_print(base_stage_name)

    if isinstance(model, SSCPPONDNFEOAgent):
        # If the model is EO, we remove the EO and evaluate the model
        _simulate_with_print("Plain NDNF (EO removed)")
        model = plain_ndnf_model
        base_stage_name = "Plain NDNF (EO removed)"

    log.info("======================================")

    # Stage 2: Prune the model
    def prune_model() -> None:
        log.info("Pruning the model...")

        sd_list = []
        prune_count = 0

        def comparison_fn(og_log, new_log):
            # Check if the action distribution is close
            # We don't use kl divergence just yet
            og_action_dist = og_log["action_distribution"]
            new_action_dist = new_log["action_distribution"]

            return torch.allclose(og_action_dist, new_action_dist, atol=1e-2)

        while True:
            log.info(f"Pruning iteration: {prune_count+1}")
            prune_result_dict = prune_neural_dnf(
                model.actor,
                simulate,
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
        prune_model()
        torch.save(model.state_dict(), model_dir / "model_mr_pruned.pth")

    post_prune_logs = _simulate_with_print(f"{base_stage_name} pruned")

    log.info("======================================")

    # 3. Thresholding
    # Different to MDP, we only threshold on the conjunctions
    og_conj_weight = model.actor.conjunctions.weights.data.clone()

    # calculate the kl divergence between two distributions p and q, where p is
    # the true distribution (action dist after pruning) and q is the estimated
    # distribution (thresholded action dist)
    def kl_divergence(p: npt.NDArray, q: npt.NDArray) -> float:
        return float(np.mean(np.where(p != 0, p * np.log(p / q), 0)))

    def threshold_model() -> tuple[float, float]:
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
            d = simulate()
            new_dist = d["action_distribution"].numpy()
            d["kl"] = kl_divergence(
                post_prune_logs["action_distribution"].numpy(), new_dist
            )
            d["threshold"] = v.item()
            result_dicts.append(d)

        # Sort the result dicts by kl divergence between the action distribution
        # after pruning and the action distribution after thresholding
        sorted_result_dicts = sorted(result_dicts, key=lambda x: x["kl"])
        best_result_dict = sorted_result_dicts[0]
        best_threshold = best_result_dict["threshold"]
        best_kl = best_result_dict["kl"]

        log.info(f"Best threshold: {best_threshold}")
        log.info(f"KL divergence: {best_result_dict['kl']}")
        model.actor.conjunctions.weights.data = (
            (torch.abs(og_conj_weight) > best_threshold)
            * torch.sign(og_conj_weight)
            * 6.0
        )
        _simulate_with_print(f"{base_stage_name} pruned thresholded")
        return best_threshold, best_kl

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
        torch.save(model.state_dict(), model_dir / "thresholded_model.pth")

    if (model_dir / "thresholded_model.pth").exists():
        thresholded_state = torch.load(
            model_dir / "thresholded_model.pth", map_location=DEVICE
        )
        model.load_state_dict(thresholded_state)
    elif (model_dir / "threshold_val_candidates.json").exists():
        with open(model_dir / "threshold_val_candidates.json", "r") as f:
            threshold_json_dict = json.load(f)
            t_val = threshold_json_dict["threshold_val"]
        apply_threshold_with_candidate(t_val)
    else:
        t_val, kl = threshold_model()
        threshold_json_dict = {}
        with open(model_dir / "threshold_val_candidates.json", "w") as f:
            threshold_json_dict["threshold_val"] = t_val
            threshold_json_dict["kl"] = kl
            json.dump(threshold_json_dict, f)

    post_threshold_log = _simulate_with_print(
        f"{base_stage_name} (thresholded)"
    )
    kl = kl_divergence(
        post_prune_logs["action_distribution"].numpy(),
        post_threshold_log["action_distribution"].numpy(),
    )
    log.info(f"KL divergence cmp to after prune: {kl}")

    log.info(model.actor.conjunctions.weights.data)
    log.info(model.actor.disjunctions.weights.data)

    log.info("======================================")

    # Stage 4. Second prune after thresholding
    # Again, check for checkpoints first
    # If the model is already pruned, then we load the pruned model
    # Otherwise, we prune the model and save the pruned model
    if (model_dir / "model_2nd_mr_pruned.pth").exists():
        pruned_state = torch.load(
            model_dir / "model_2nd_mr_pruned.pth", map_location=DEVICE
        )
        model.load_state_dict(pruned_state)
    else:
        prune_model()
        torch.save(model.state_dict(), model_dir / "model_2nd_mr_pruned.pth")

    second_prune_logs = _simulate_with_print(f"{base_stage_name} 2nd prune")
    kl = kl_divergence(
        post_prune_logs["action_distribution"].numpy(),
        second_prune_logs["action_distribution"].numpy(),
    )
    log.info(f"KL divergence cmp to after 1st prune: {kl}")
    with open(model_dir / "final_model_state_dict_log.log", "w") as f:
        return_per_episode = synthesize(second_prune_logs["return_per_episode"])
        f.write(f"Avg. return: {return_per_episode['mean']}\n")
        f.write(
            f"Action distribution: {second_prune_logs['action_distribution']}\n"
        )
        f.write(str(model.actor.state_dict()))
        f.write("\n")
        f.write(f"KL divergence cmp to after 1st prune: {kl}\n")


def post_train_eval(eval_cfg: DictConfig):
    experiment_name = eval_cfg["experiment_name"]
    use_ndnf = "ndnf" in experiment_name
    assert use_ndnf

    use_state_no_as_obs = "sn" in experiment_name
    assert not use_state_no_as_obs, "Has to use wall status as observation"

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
        post_training(
            model_dir=model_dir,
            model=model,
            envs=envs,
            process_obs=process_obs,
        )

        log.info("======================================\n")


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
