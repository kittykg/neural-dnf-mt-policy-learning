from datetime import datetime
import logging
from pathlib import Path
import random
import traceback
from typing import Any

import gymnasium as gym
import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import DictConfig
import torch
from torch.utils.tensorboard import SummaryWriter  # type: ignore
import wandb

from corridor_grid.envs import DoorCorridorEnv
from door_corridor_ppo import (
    train_ppo,
    make_env,
    DCPPONDNFMutexTanhAgent,
)
from utils import post_to_discord_webhook


log = logging.getLogger()
single_env = DoorCorridorEnv(render_mode="rgb_array")

# There are 5 relevant observations, check
# `plots/unique_relevant_observations.png` for the visualization.
obs, _ = single_env.reset()
RELEVANT_OBSERVATIONS = [obs]
for i, a in enumerate([1, 3, 2, 3, 2, 3, 2, 2]):
    obs, _, _, _, _ = single_env.step(a)
    if (i + 1) in [1, 2, 6, 7]:
        RELEVANT_OBSERVATIONS.append(obs)


def single_eval(
    model: DCPPONDNFMutexTanhAgent,
    device: torch.device,
    discretise_img_encoding: bool = False,
    num_episodes: int = 100,
) -> dict[str, Any]:
    log.info("Evaluating model")

    model.to(device)
    model.eval()

    num_processes = 8
    envs = gym.vector.SyncVectorEnv(
        [make_env(i, i, False) for i in range(num_processes)]
    )
    logs = {
        "num_frames_per_episode": [],
        "return_per_episode": [],
        "mutual_exclusivity": True,
        "missing_actions": False,
    }
    next_obs_dict, _ = envs.reset()
    next_obs = torch.Tensor(next_obs_dict["image"]).to(device)
    next_obs_dict = {"image": next_obs}

    log_done_counter = 0
    log_episode_return = torch.zeros(num_processes, device=device)
    log_episode_num_frames = torch.zeros(num_processes, device=device)

    while log_done_counter < num_episodes:
        with torch.no_grad():
            actions = model.get_actions(
                preprocessed_obs=next_obs_dict,
                use_argmax=True,
                discretise_img_encoding=discretise_img_encoding,
            )
        tanh_action = np.count_nonzero(actions[1] > 0, axis=1)
        if np.any(tanh_action > 1):
            logs["mutual_exclusivity"] = False
        if np.any(tanh_action == 0):
            logs["missing_actions"] = True
        next_obs_dict, reward, terminations, truncations, infos = envs.step(
            actions[0]
        )
        next_obs = torch.Tensor(next_obs_dict["image"]).to(device)
        next_obs_dict = {"image": next_obs}
        next_done = np.logical_or(terminations, truncations)

        log_episode_return += torch.tensor(
            reward, device=device, dtype=torch.float
        )
        log_episode_num_frames += torch.ones(num_processes, device=device)

        for i, d in enumerate(next_done):
            if d:
                log_done_counter += 1
                logs["return_per_episode"].append(log_episode_return[i].item())
                logs["num_frames_per_episode"].append(
                    log_episode_num_frames[i].item()
                )

        mask = 1 - torch.tensor(next_done, device=device, dtype=torch.float)
        log_episode_return *= mask
        log_episode_num_frames *= mask

    # Get action distributions
    with torch.no_grad():
        obs_dict = {
            "image": torch.tensor(
                np.array([d["image"] for d in RELEVANT_OBSERVATIONS])
            )
            .float()
            .to(device)
        }
        aux_loss_dict = model.get_aux_loss(obs_dict)
        l_emb_dis = aux_loss_dict["l_emb_dis"]
        l_disj_l1_mod = aux_loss_dict["l_disj_l1_mod"]
        l_tanh_conj = aux_loss_dict["l_tanh_conj"]

        logs["l_emb_dis"] = l_emb_dis.item()
        logs["l_disj_l1_mod"] = l_disj_l1_mod.item()
        logs["l_tanh_conj"] = l_tanh_conj.item()
        l_mt_ce2 = aux_loss_dict["l_mt_ce2"]
        logs["l_mt_ce2"] = l_mt_ce2.item()

    return logs


def compute_property_value(log: dict[str, Any]):
    # Also encode the "mutual_exclusivity" and "missing_actions" as binary bits
    # then convert to decimal
    # The bits is in order of "mutual_exclusivity" and "missing_actions"
    # The ideal value is 0b10 which is 2
    me_bit = 1 if log["mutual_exclusivity"] else 0
    ma_bit = 1 if log["missing_actions"] else 0
    return int(f"0b{me_bit}{ma_bit}", 2)


def train_ppo_wrapper(cfg: DictConfig):
    # Randomly select a seed based on the current time
    ts = datetime.now().timestamp()
    random.seed(ts)
    seed = random.randrange(10000)

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    training_cfg = cfg["training"]

    run = wandb.init(dir=HydraConfig.get().run.dir, sync_tensorboard=True)
    assert run is not None, "Wandb run is not initialized"

    use_ndnf = "ndnf" in training_cfg["experiment_name"]
    assert use_ndnf, "Must be NDNF based experiment"

    use_mt = training_cfg["use_mt"]
    assert use_mt, "Must be MT based experiment"

    full_experiment_name = f"{training_cfg['experiment_name']}_{seed}_{int(ts)}"

    # Override the image encoder parameters
    training_cfg["customised_image_encoder"][
        "kernel_size"
    ] = wandb.config.kernel_size
    training_cfg["customised_image_encoder"][
        "encoder_output_chanel"
    ] = wandb.config.eoc
    training_cfg["customised_image_encoder"][
        "extra_layer_out"
    ] = wandb.config.extra_layer_out

    # Override PPO parameters
    training_cfg["learning_rate"] = wandb.config.learning_rate
    training_cfg["num_envs"] = wandb.config.num_envs
    training_cfg["num_steps"] = wandb.config.num_steps
    training_cfg["num_minibatches"] = wandb.config.num_minibatches
    training_cfg["update_epochs"] = wandb.config.update_epochs
    training_cfg["clip_coef"] = wandb.config.clip_coef
    training_cfg["gae_lambda"] = wandb.config.gae_lambda
    training_cfg["vf_coef"] = wandb.config.vf_coef
    training_cfg["ent_coef"] = wandb.config.ent_coef

    # OVerride delta scheduler parameters
    training_cfg["dds"]["delta_decay_delay"] = int(
        wandb.config.delta_decay_delay
    )
    training_cfg["dds"]["delta_decay_steps"] = int(
        wandb.config.delta_decay_steps
    )

    # Override the lambda parameters
    training_cfg["aux_loss"]["emb_dis_lambda"] = wandb.config.emb_dis_lambda
    training_cfg["aux_loss"][
        "dis_l1_mod_lambda"
    ] = wandb.config.dis_l1_mod_lambda
    training_cfg["aux_loss"]["tanh_conj_lambda"] = wandb.config.tanh_conj_lambda
    training_cfg["aux_loss"]["mt_ce2_lambda"] = wandb.config.mt_ce2_lambda

    use_discord_webhook = cfg["webhook"]["use_discord_webhook"]
    msg_body = None
    errored = False

    try:
        writer_dir = Path(HydraConfig.get().run.dir) / "tb"
        writer = SummaryWriter(writer_dir)
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % (
                "\n".join(
                    [
                        f"|{key}|{value}|"
                        for key, value in vars(training_cfg).items()
                    ]
                )
            ),
        )

        _, model = train_ppo(
            training_cfg,
            training_cfg["experiment_name"],
            writer,
            save_model=False,
        )

        assert isinstance(model, DCPPONDNFMutexTanhAgent)
        torch.save(model.state_dict(), "model.pt")
        model.actor.set_delta_val(1.0)

        device = torch.device("cpu")

        ndnf_mt_log = single_eval(model, device, False)
        ndnf_mt_dis_log = single_eval(model, device, True)
        ndnf_mt_dis_return = np.array(
            ndnf_mt_dis_log["return_per_episode"]
        ).mean()  # in range [-270, -8]
        # All the aux loss are in range [0, +inf]
        ndnf_mt_dis_l_emb_dis = ndnf_mt_dis_log["l_emb_dis"]
        ndnf_mt_dis_l_disj_l1_mod = ndnf_mt_dis_log["l_disj_l1_mod"]
        ndnf_mt_dis_l_tanh_conj = ndnf_mt_dis_log["l_tanh_conj"]
        ndnf_mt_dis_l_mt_ce2 = ndnf_mt_dis_log["l_mt_ce2"]

        combined_metric = ndnf_mt_dis_return - 0.1 * (
            ndnf_mt_dis_l_emb_dis
            + ndnf_mt_dis_l_disj_l1_mod
            + ndnf_mt_dis_l_tanh_conj
            + ndnf_mt_dis_l_mt_ce2
        )

        wandb.log(
            {
                "ndnf_mt_return_per_episode": np.array(
                    ndnf_mt_log["return_per_episode"]
                ).mean(),
                "ndnf_mt_dis_return_per_episode": ndnf_mt_dis_return,
                "ndnf_mt_dis_l_emb_dis": ndnf_mt_dis_l_emb_dis,
                "ndnf_mt_dis_l_disj_l1_mod": ndnf_mt_dis_l_disj_l1_mod,
                "ndnf_mt_dis_l_tanh_conj": ndnf_mt_dis_l_tanh_conj,
                "ndnf_mt_dis_l_mt_ce2": ndnf_mt_dis_l_mt_ce2,
                "combined_metric": combined_metric,
                "ndnf_mt_dis_property_value": compute_property_value(
                    ndnf_mt_dis_log
                ),
            }
        )

        if use_discord_webhook:
            msg_body = "Success!"
    except BaseException as e:
        if use_discord_webhook:
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
                experiment_name=full_experiment_name,
                message_body=msg_body,
                errored=errored,
            )
            wandb.finish()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    use_wandb = cfg["wandb"]["use_wandb"]
    assert use_wandb, "Must use wandb for hyperparameter search"

    train_ppo_wrapper(cfg)


if __name__ == "__main__":
    import multiprocessing as mp

    if mp.get_start_method() != "fork":
        mp.set_start_method("fork", force=True)

    run_experiment()
