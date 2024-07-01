from collections import namedtuple, defaultdict
import logging
import math
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
from utils import post_to_discord_webhook


Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward")
)
ObservationType = tuple[int, int, int]
steps_done = 0
log = logging.getLogger()
N_ACTIONS = 2
EVAL_ENV_SEED = 1


class TabularQAgent:
    n_actions: int
    q_table: defaultdict[ObservationType, npt.NDArray[np.float64]]

    gamma: float
    alpha: float

    steps_done: int
    eps_end: float
    eps_start: float
    eps_decay: float

    use_sarsa: bool

    td_errors: list[float]

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
        super().__init__()

        self.n_actions = n_actions
        self.q_table = defaultdict(lambda: np.zeros(N_ACTIONS))

        self.gamma = gamma
        self.alpha = alpha

        self.steps_done = 0
        self.eps_end = eps_end
        self.eps_start = eps_start
        self.eps_decay = eps_decay

        self.use_sarsa = use_sarsa

        self.td_errors = []

    def best_value_and_action(self, obs: ObservationType) -> tuple[float, int]:
        action_values = self.q_table[obs]
        best_value = np.max(action_values)
        best_action = np.argmax(action_values)
        return best_value, best_action  # type: ignore

    def select_epsilon_greedy_action(
        self, env: BlackjackEnv, obs: ObservationType
    ) -> tuple[int, float]:
        sample = random.random()
        eps_threshold = self.eps_end + (
            self.eps_start - self.eps_end
        ) * math.exp(-1.0 * self.steps_done / self.eps_decay)
        self.steps_done += 1
        action = None
        if sample > eps_threshold:
            action = np.argmax(self.q_table[obs])
        else:
            action = env.action_space.sample()
        return int(action), eps_threshold

    def simulate_one_episode(
        self, env: BlackjackEnv
    ) -> tuple[float, int, float]:
        if self.use_sarsa:
            return self._simulate_one_episode_sarsa(env)
        else:
            return self._simulate_one_episode_q_learning(env)

    def _simulate_one_episode_sarsa(
        self, env: BlackjackEnv
    ) -> tuple[float, int, float]:
        total_reward = 0
        obs, _ = env.reset()
        action, _ = self.select_epsilon_greedy_action(env, obs)
        terminated = False
        truncated = False
        episode_duration = 0
        eps_threshold = 0

        while not terminated and not truncated:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            # Update Q table
            # SARSA: Q(s,a) = Q(s,a) + alpha * (r + gamma * Q(s',a') - Q(s,a))
            new_action, eps_threshold = self.select_epsilon_greedy_action(
                env, next_obs
            )
            new_value = reward + self.gamma * self.q_table[next_obs][new_action]
            old_value = self.q_table[obs][action]
            self.q_table[obs][action] = old_value + self.alpha * (
                new_value - old_value
            )

            self.td_errors.append(new_value - old_value)

            obs = next_obs
            action = new_action
            episode_duration += 1

        return total_reward, episode_duration, eps_threshold

    def _simulate_one_episode_q_learning(
        self, env: BlackjackEnv
    ) -> tuple[float, int, float]:
        total_reward = 0
        obs, _ = env.reset()
        terminated = False
        truncated = False
        episode_duration = 0
        eps_threshold = 0

        while not terminated and not truncated:
            action, eps_threshold = self.select_epsilon_greedy_action(env, obs)
            new_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            # Update Q table
            # Q learning: Q(s,a) = Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
            best_value, _ = self.best_value_and_action(new_obs)
            new_value = reward + self.gamma * best_value
            old_value = self.q_table[obs][action]
            self.q_table[obs][action] = old_value + self.alpha * (
                new_value - old_value
            )

            self.td_errors.append(new_value - old_value)

            obs = new_obs
            episode_duration += 1

        return total_reward, episode_duration, eps_threshold


def train(
    env: BlackjackEnv,
    agent: TabularQAgent,
    num_episodes: int,
    use_wandb: bool,
    full_experiment_name: str,
    logging_freq: int = 1000,
    save_freq: int = 1000,
    plot_policy: bool = True,
) -> None:
    def get_moving_average_for_plot(
        data: list[float], rolling_length: int = 500
    ) -> npt.NDArray[np.float64]:
        t = np.array(data, dtype=np.float64)
        return (
            np.convolve(t, np.ones(rolling_length), mode="valid")
            / rolling_length
        )

    episode_rewards = []
    episode_durations = []
    for i in range(1, int(num_episodes) + 1):
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
                log_dict = {
                    "episode": i,
                    "eps_threshold": eps_threshold,
                    "total_reward": reward_moving_average[-1],
                    "duration": episode_duration_moving_average[-1],
                    "td_error": get_moving_average_for_plot(agent.td_errors)[
                        -1
                    ],
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
    plot = create_target_policy_plots(
        target_policy,
        model_name,
        argmax=True,
    )
    plot.savefig(f"{model_name}_argmax_policy.png")

    if use_wandb:
        wandb.log(
            {"argmax_policy": wandb.Image(f"{model_name}_argmax_policy.png")}
        )

    plt.close()
    plot = create_target_policy_plots(
        target_policy,
        model_name,
        argmax=False,
    )
    plot.savefig(f"{model_name}_soft_policy.png")

    if use_wandb:
        wandb.log({"soft_policy": wandb.Image(f"{model_name}_soft_policy.png")})

    plt.close()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    training_cfg = cfg["training"]
    seed = training_cfg["seed"]

    if seed is None:
        seed = random.randint(0, 1000000)

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
    agent = TabularQAgent(
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
            training_cfg["num_episodes"],
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
