import random
from datetime import datetime
from pathlib import Path
import traceback


import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import DictConfig
import torch
from torch.utils.tensorboard import SummaryWriter  # type: ignore
import wandb


from blackjack_ppo import train_ppo
from utils import post_to_discord_webhook


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

    use_ndnf = "ndnf" in training_cfg["experiment_name"]
    assert use_ndnf, "Must be NDNF based experiment"

    use_mt = "mt" in training_cfg["experiment_name"]

    # Model architecture
    training_cfg["model_latent_size"] = wandb.config.model_latent_size

    # Override the PPO parameters
    training_cfg["learning_rate"] = wandb.config.learning_rate
    training_cfg["num_envs"] = wandb.config.num_envs
    training_cfg["num_steps"] = wandb.config.num_steps
    training_cfg["num_minibatches"] = wandb.config.num_minibatches
    training_cfg["update_epochs"] = wandb.config.update_epochs
    training_cfg["clip_coef"] = wandb.config.clip_coef
    training_cfg["vf_coef"] = wandb.config.vf_coef
    training_cfg["ent_coef"] = wandb.config.ent_coef

    # Override the lambda parameter with the one from the sweep
    training_cfg["dds"]["delta_decay_delay"] = int(
        wandb.config.delta_decay_delay
    )
    training_cfg["dds"]["delta_decay_steps"] = int(
        wandb.config.delta_decay_steps
    )

    # Override the auxiliary loss related parameters
    training_cfg["aux_loss"]["delta_one_delay"] = int(
        wandb.config.delta_one_delay
    )
    training_cfg["aux_loss"][
        "dis_l1_mod_lambda"
    ] = wandb.config.dis_l1_mod_lambda

    if use_mt:
        training_cfg["aux_loss"]["mt_ce2_lambda"] = wandb.config.mt_ce2_lambda

    full_experiment_name = f"{training_cfg['experiment_name']}_{int(ts)}"

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

        _, _, eval_log = train_ppo(
            training_cfg, full_experiment_name, True, writer, save_model=True
        )
        assert eval_log is not None

        policy_error_cmp_to_q = eval_log["policy_error_cmp_to_q"]
        mutual_exclusivity = eval_log["mutual_exclusivity"]
        missing_actions = eval_log["missing_actions"]

        # Aim to minimise combined metric
        combined_metric = policy_error_cmp_to_q
        if not mutual_exclusivity:
            # Mutual exclusivity is not satisfied
            combined_metric += eval_log["mutual_exclusivity_violations_count"]

        if missing_actions:
            # There are missing actions
            combined_metric += eval_log["missing_actions_count"]

        wandb.log({"combined_metric": combined_metric})

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
    # torch.autograd.set_detect_anomaly(True)  # type: ignore

    use_wandb = cfg["wandb"]["use_wandb"]
    assert use_wandb, "Must use wandb for hyperparameter search"

    train_ppo_wrapper(cfg)


if __name__ == "__main__":
    import multiprocessing as mp

    if mp.get_start_method() != "fork":
        mp.set_start_method("fork", force=True)

    run_experiment()
