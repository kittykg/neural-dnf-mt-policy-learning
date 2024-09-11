import math
import random
from typing import Any

from gymnasium import Env
import numpy as np
import numpy.typing as npt


class TabularQAgent:
    q_table: npt.NDArray[np.float64] | dict[Any, npt.NDArray[np.float64]]

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
        gamma: float,
        alpha: float,
        eps_end: float,
        eps_start: float,
        eps_decay: float,
        use_sarsa: bool = False,
    ) -> None:
        self.gamma = gamma
        self.alpha = alpha

        self.steps_done = 0
        self.eps_end = eps_end
        self.eps_start = eps_start
        self.eps_decay = eps_decay

        self.use_sarsa = use_sarsa

        self.td_errors = []

    def _obs_processing(self, obs: Any) -> Any:
        return obs

    def best_value_and_action(self, obs: Any) -> tuple[float, int]:
        action_values = self.q_table[obs]
        best_value = np.max(action_values)
        best_action = np.argmax(action_values)
        return best_value, best_action  # type: ignore

    def select_epsilon_greedy_action(
        self, env: Env, obs: Any
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

    def simulate_one_episode(self, env: Env) -> tuple[float, int, float]:
        if self.use_sarsa:
            return self._simulate_one_episode_sarsa(env)
        else:
            return self._simulate_one_episode_q_learning(env)

    def _simulate_one_episode_sarsa(self, env: Env) -> tuple[float, int, float]:
        total_reward = 0
        obs, _ = env.reset()
        obs = self._obs_processing(obs)
        action, _ = self.select_epsilon_greedy_action(env, obs)
        terminated = False
        truncated = False
        episode_duration = 0
        eps_threshold = 0

        while not terminated and not truncated:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_obs = self._obs_processing(next_obs)
            total_reward += reward  # type: ignore

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
        self, env: Env
    ) -> tuple[float, int, float]:
        total_reward = 0
        obs, _ = env.reset()
        obs = self._obs_processing(obs)
        terminated = False
        truncated = False
        episode_duration = 0
        eps_threshold = 0

        while not terminated and not truncated:
            action, eps_threshold = self.select_epsilon_greedy_action(env, obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_obs = self._obs_processing(next_obs)
            total_reward += reward  # type: ignore

            # Update Q table
            # Q learning: Q(s,a) = Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
            best_value, _ = self.best_value_and_action(next_obs)
            new_value = reward + self.gamma * best_value  # type: ignore
            old_value = self.q_table[obs][action]
            self.q_table[obs][action] = old_value + self.alpha * (
                new_value - old_value
            )

            self.td_errors.append(new_value - old_value)

            obs = next_obs
            episode_duration += 1

        return total_reward, episode_duration, eps_threshold


def get_moving_average_for_plot(
    data: list[float], rolling_length: int = 500
) -> npt.NDArray[np.float64]:
    t = np.array(data, dtype=np.float64)
    return (
        np.convolve(t, np.ones(rolling_length), mode="valid") / rolling_length
    )
