import logging
import math
from pathlib import Path
import random
import traceback

import hydra
from hydra.core.hydra_config import HydraConfig
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import wandb

import gymnasium as gym
from gymnasium import Env
from gymnasium.wrappers.record_video import RecordVideo
from utils import post_to_discord_webhook

log = logging.getLogger()
EVAL_ENV_SEED = 123


class TabularQAgent:
    n_states: int
    n_actions: int
    q_table: npt.NDArray[np.float64]

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
        n_states: int,
        n_actions: int,
        gamma: float,
        alpha: float,
        eps_end: float,
        eps_start: float,
        eps_decay: float,
        use_sarsa: bool = False,
    ) -> None:
        super().__init__()

        self.n_states = n_states
        self.n_actions = n_actions
        self.q_table = np.zeros((n_states, n_actions))

        self.gamma = gamma
        self.alpha = alpha

        self.steps_done = 0
        self.eps_end = eps_end
        self.eps_start = eps_start
        self.eps_decay = eps_decay

        self.use_sarsa = use_sarsa

        self.td_errors = []

    def best_value_and_action(self, state: int) -> tuple[float, int]:
        action_values = self.q_table[state, :]
        best_value = np.max(action_values)
        best_action = np.argmax(action_values)
        return best_value, best_action  # type: ignore

    def select_epsilon_greedy_action(
        self, env: Env, state: int, eval: bool = False
    ) -> tuple[int, float]:
        sample = random.random()
        eps_threshold = self.eps_end + (
            self.eps_start - self.eps_end
        ) * math.exp(-1.0 * self.steps_done / self.eps_decay)
        self.steps_done += 1
        action = None
        if eval:
            action = np.argmax(self.q_table[state, :])
        elif sample > eps_threshold:
            action = np.argmax(self.q_table[state, :])
        else:
            action = env.action_space.sample()
        return int(action), eps_threshold

    def simulate_one_episode(self, env: Env) -> tuple[float, int, float]:
        if self.use_sarsa:
            return self._simulate_one_episode_sarsa(env)
        else:
            return self._simulate_one_episode_q_learning(env)

    def _simulate_one_episode_sarsa(self, env: Env) -> tuple[float, int, float]:
        total_reward = 0
        state, _ = env.reset()
        action, _ = self.select_epsilon_greedy_action(env, state)
        terminated = False
        truncated = False
        episode_duration = 0
        eps_threshold = 0

        while not terminated and not truncated:
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward  # type: ignore

            # Update Q table
            # SARSA: Q(s,a) = Q(s,a) + alpha * (r + gamma * Q(s',a') - Q(s,a))
            new_action, eps_threshold = self.select_epsilon_greedy_action(
                env, next_state
            )
            new_value = (
                reward + self.gamma * self.q_table[next_state, new_action]
            )
            old_value = self.q_table[state, action]
            self.q_table[state, action] = old_value + self.alpha * (
                new_value - old_value
            )

            self.td_errors.append(new_value - old_value)

            state = next_state
            action = new_action
            episode_duration += 1

        return total_reward, episode_duration, eps_threshold

    def _simulate_one_episode_q_learning(
        self, env: Env
    ) -> tuple[float, int, float]:
        total_reward = 0
        state, _ = env.reset()
        terminated = False
        truncated = False
        episode_duration = 0
        eps_threshold = 0

        while not terminated and not truncated:
            action, eps_threshold = self.select_epsilon_greedy_action(
                env, state
            )
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward  # type: ignore

            # Update Q table
            # Q learning: Q(s,a) = Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
            best_value, _ = self.best_value_and_action(next_state)
            new_value = reward + self.gamma * best_value  # type: ignore
            old_value = self.q_table[state, action]
            self.q_table[state, action] = old_value + self.alpha * (
                new_value - old_value
            )

            self.td_errors.append(new_value - old_value)

            state = next_state
            episode_duration += 1

        return total_reward, episode_duration, eps_threshold

    def after_train_evaluate(self) -> tuple[float, float]:
        env = gym.make("Taxi-v3", render_mode="rgb_array")
        env = RecordVideo(env, video_folder="videos")

        if self.use_sarsa:
            total_reward, episode_duration = self._evaluate_sarsa(env)  # type: ignore
        else:
            total_reward, episode_duration = self._evaluate_q_learning(env)  # type: ignore
        env.close()

        return total_reward, episode_duration

    def _evaluate_sarsa(
        self, env: Env, eval_env_seed: int | None = None
    ) -> tuple[float, int]:
        total_reward = 0
        state, _ = env.reset(seed=eval_env_seed)
        action, _ = self.select_epsilon_greedy_action(env, state)
        terminated = False
        truncated = False
        episode_duration = 0

        while not terminated and not truncated:
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward  # type: ignore

            new_action, _ = self.select_epsilon_greedy_action(
                env, next_state, eval=True
            )

            state = next_state
            action = new_action
            episode_duration += 1

        return total_reward, episode_duration

    def _evaluate_q_learning(
        self, env: Env, eval_env_seed: int | None = None
    ) -> tuple[float, int]:
        total_reward = 0
        state, _ = env.reset(seed=eval_env_seed)
        terminated = False
        truncated = False
        episode_duration = 0

        while not terminated and not truncated:
            action, _ = self.select_epsilon_greedy_action(env, state, eval=True)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward  # type: ignore
            episode_duration += 1

        return total_reward, episode_duration


def train(
    env: Env,
    agent: TabularQAgent,
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

    log.info("Table:")
    log.info("\t" + "\t".join([str(i) for i in range(agent.n_actions)]))
    for i in range(agent.n_states):
        info_str = f"S{i}\t"
        info_str += "\t".join(
            [f"{agent.q_table[i, j]:.2f}" for j in range(agent.n_actions)]
        )
        log.info(f"{info_str}")


def after_train_eval(agent: TabularQAgent, use_wandb: bool):
    # Evaluate the agent
    total_reward, episode_duration = agent.after_train_evaluate()

    log.info(f"Evaluated agent with total reward {total_reward}")
    log.info(f"Evaluated agent with episode duration {episode_duration}")

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
        seed = random.randint(0, 1000000)

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
    n_states = 500
    agent = TabularQAgent(
        n_states=n_states,
        n_actions=6,
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
