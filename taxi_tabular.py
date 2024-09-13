import logging
from pathlib import Path
import random
import traceback

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import wandb

import gymnasium as gym
from gymnasium import Env
from utils import post_to_discord_webhook

from eval.taxi_tabular_multirun_eval import simulate_on_env, record_video
from tabular_common import TabularQAgent

log = logging.getLogger()
EVAL_ENV_SEED = 123
N_ACTIONS = 6
N_STATES = 500


class TaxiTabularQAgent(TabularQAgent):
    n_states: int
    n_actions: int
    q_table: npt.NDArray[np.float64]

    def __init__(
        self,
        n_states: int,
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

        self.n_states = n_states
        self.n_actions = n_actions
        self.q_table = np.zeros((n_states, n_actions))


def train(
    env: Env,
    agent: TaxiTabularQAgent,
    num_episodes: int,
    use_wandb: bool,
    logging_freq: int = 1000,
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
            log_dict = {
                "episode": i,
                "eps_threshold": eps_threshold,
                "total_reward": total_reward,
                "duration": episode_duration,
                "td_error": float(np.mean(agent.td_errors)),
            }
            wandb.log(log_dict)

    log.info("Training complete")


def after_train_eval(agent: TaxiTabularQAgent, use_wandb: bool):
    # Evaluate the agent
    env = gym.make("Taxi-v3")
    ret_log = simulate_on_env(
        agent.q_table, env, use_argmax=True, epsilon=0.0, num_episodes=100
    )

    log.info(f"Evaluated agent")
    log.info(f"Average return: {ret_log['avg_return']:.3f}")
    log.info(f"Standard deviation of return: {ret_log['std_return']:.3f}")
    log.info(f"Standard error of return: {ret_log['std_error']:.3f}")

    record_video(
        agent.q_table, use_argmax=True, epsilon=0.0, video_dir=Path("videos")
    )

    if use_wandb:
        wandb.log(
            {
                "video": wandb.Video(
                    "videos/rl-video-episode-0.mp4", format="mp4"
                )
            }
        )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    training_cfg = cfg["training"]
    seed = training_cfg["seed"]

    if seed is None:
        seed = random.randint(0, 10000)

    # Expect the experiment name to be in the format of
    # taxi_tab_..._..._..._...
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

    env = gym.make("Taxi-v3", render_mode="ansi")
    agent = TaxiTabularQAgent(
        n_states=N_STATES,
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
            logging_freq=training_cfg["logging_freq"],
        )

        after_train_eval(agent, use_wandb)

        df = pd.DataFrame(agent.q_table)
        df.to_csv(f"{full_experiment_name}.csv", index=False)
        np.save(f"{full_experiment_name}.npy", agent.q_table)

        if use_wandb:
            columns = [f"S{i}" for i in range(agent.n_states)]
            table = []
            for j in range(agent.n_actions):
                table.append(
                    [agent.q_table[i, j] for i in range(agent.n_states)]
                )
            wandb.log({"Q-table": wandb.Table(data=table, columns=columns)})

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
        env.close()
        if not errored:
            path = Path(HydraConfig.get().run.dir)
            path.rename(path.absolute().parent / run_dir_name)


if __name__ == "__main__":
    run_experiment()
