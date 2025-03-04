from collections import defaultdict
import logging
from pathlib import Path
import random
import traceback

import gymnasium as gym
from gymnasium.envs.toy_text.blackjack import BlackjackEnv
import hydra
from hydra.core.hydra_config import HydraConfig
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import wandb

from blackjack_common import get_target_policy, create_target_policy_plots
from tabular_common import TabularQAgent, get_moving_average_for_plot
from utils import post_to_discord_webhook


EVAL_ENV_SEED = 1
N_ACTIONS = 2
ObservationType = tuple[int, int, int]

log = logging.getLogger()


class BlackjackTabularQAgent(TabularQAgent):
    n_actions: int
    q_table: defaultdict[ObservationType, npt.NDArray[np.float64]]

    def __init__(
        self,
        n_actions: int,
        gamma: float,
        alpha: float,
        eps_end: float,
        eps_start: float,
        eps_decay: float,
        use_sarsa: bool = False,
    ) -> None:
        super().__init__(
            gamma=gamma,
            alpha=alpha,
            eps_end=eps_end,
            eps_start=eps_start,
            eps_decay=eps_decay,
            use_sarsa=use_sarsa,
        )

        self.n_actions = n_actions
        self.q_table = defaultdict(lambda: np.zeros(N_ACTIONS))


def train(
    env: BlackjackEnv,
    agent: BlackjackTabularQAgent,
    num_episodes: int,
    use_wandb: bool,
    full_experiment_name: str,
    logging_freq: int = 1000,
    save_freq: int = 1000,
    plot_policy: bool = True,
) -> None:
    episode_rewards = []
    episode_durations = []
    for i in range(1, num_episodes + 1):
        # Initialize the environment and get it's state
        (
            total_reward,
            episode_duration,
            eps_threshold,
        ) = agent.simulate_one_episode(env)

        episode_rewards.append(total_reward)
        episode_durations.append(episode_duration)

        if i % logging_freq == 0:
            log.info(
                f"Episode {i}\t"
                f"Epsilon threshold: {eps_threshold:.2f}\t"
                f"Reward: {total_reward}\t"
                f"Duration: {episode_duration}"
            )

            if use_wandb:
                # Plot the moving average of the rewards and durations
                reward_moving_average = get_moving_average_for_plot(
                    episode_rewards
                )
                episode_duration_moving_average = get_moving_average_for_plot(
                    episode_durations
                )
                td_error_moving_average = get_moving_average_for_plot(
                    agent.td_errors
                )
                log_dict = {
                    "episode": i,
                    "eps_threshold": eps_threshold,
                    "total_reward": reward_moving_average[-1],
                    "duration": episode_duration_moving_average[-1],
                    "td_error": td_error_moving_average[-1],
                }
                wandb.log(log_dict)

        if i % save_freq == 0:
            table_name = f"{full_experiment_name}_{i}.csv"
            df = pd.DataFrame(agent.q_table)
            df.to_csv(table_name)

    log.info("Training complete")

    log.info("Table:")
    log.info("\t" + "\t".join([str(i) for i in range(agent.n_actions)]))
    df = pd.DataFrame(agent.q_table)
    log.info(df)

    if plot_policy:
        plot_policy_grid_after_train(
            Path(table_name), full_experiment_name, use_wandb
        )


def plot_policy_grid_after_train(
    csv_path: Path, model_name: str, use_wandb: bool
):
    target_policy = get_target_policy(csv_path)
    plot = create_target_policy_plots(target_policy, model_name)
    plot.savefig(f"{model_name}_argmax_policy.png")

    if use_wandb:
        wandb.log(
            {"argmax_policy": wandb.Image(f"{model_name}_argmax_policy.png")}
        )

    plt.close()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    training_cfg = cfg["training"]
    seed = training_cfg["seed"]

    if seed is None:
        seed = random.randint(0, 10000)

    # Expect the experiment name to be in the format of
    # blackjack_tab_..._..._..._...
    name_list = training_cfg["experiment_name"].split("_")
    # Insert "sarsa" or "q" after "tab"
    name_list.insert(2, "sarsa" if training_cfg["use_sarsa"] else "q")
    # Add the seed at the end of the name list
    name_list.append(str(seed))
    full_experiment_name = "_".join(name_list)

    log.info(f"Experiment {full_experiment_name} started.")

    # Set random seed
    np.random.seed(seed)
    random.seed(seed)

    run_dir_name = "-".join(
        [
            (s.upper() if i in [0, 1] else s)
            for i, s in enumerate(full_experiment_name.split("_"))
        ]
    )

    use_wandb = cfg["wandb"]["use_wandb"]
    if use_wandb:
        run = wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"]["entity"],
            config=OmegaConf.to_container(
                training_cfg, resolve=True, throw_on_missing=True
            ),  # type: ignore
            dir=HydraConfig.get().run.dir,
            name=run_dir_name,
            tags=cfg["wandb"]["tags"] if "tags" in cfg["wandb"] else [],
            group=cfg["wandb"]["group"] if "group" in cfg["wandb"] else None,
        )

    env: BlackjackEnv = gym.make("Blackjack-v1", render_mode="rgb_array")  # type: ignore
    agent = BlackjackTabularQAgent(
        n_actions=N_ACTIONS,
        gamma=training_cfg["gamma"],
        alpha=training_cfg["alpha"],
        eps_end=training_cfg["eps_end"],
        eps_start=training_cfg["eps_start"],
        eps_decay=training_cfg["eps_decay"],
        use_sarsa=training_cfg["use_sarsa"],
    )

    use_discord_webhook = cfg["webhook"]["use_discord_webhook"]
    msg_body = None
    keyboard_interrupt = None
    errored = False

    try:
        train(
            env,
            agent,
            int(training_cfg["num_episodes"]),
            use_wandb,
            full_experiment_name,
            logging_freq=training_cfg["logging_freq"],
            save_freq=training_cfg["save_freq"],
        )

        table_name = f"{full_experiment_name}.csv"
        df = pd.DataFrame(agent.q_table)
        df.to_csv(table_name)

        if use_wandb:
            wandb.save(glob_str=table_name)

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
                experiment_name=full_experiment_name,
                message_body=msg_body,
                errored=errored,
                keyboard_interrupt=keyboard_interrupt,
            )
        if use_wandb:
            wandb.finish()
        if not errored:
            path = Path(HydraConfig.get().run.dir)
            path.rename(path.absolute().parent / run_dir_name)


if __name__ == "__main__":
    run_experiment()
