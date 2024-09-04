from typing import Any, Callable

from gymnasium import Env
import numpy as np
from problog.program import PrologString
from problog import get_evaluatable


def prolog_inference_in_env_single_run(
    env: Env,
    problog_rules: list[str],
    num_actions: int,
    context_problog_gen_fn: Callable[[Any], list[str]],
    use_argmax: bool = False,
) -> dict[str, Any]:
    """
    Run a single episode of the environment with the given problog rules and
    `context_problog_gen_fn`. The `context_problog_gen_fn` should take an
    observation and return a list of strings that represent the context of the
    observation in problog format: e.g. ["1::input(0).", "0::input(1)."].
    Return the episode reward and the trace of the episode, where the trace is a
    list of tuples (observation, action, action_probs).
    """
    obs, _ = env.reset()
    terminated, truncated = False, False
    episode_reward = 0
    num_frames = 0

    trace: list[tuple[tuple, int, np.ndarray]] = []

    while not terminated and not truncated:
        context_problog = context_problog_gen_fn(obs)

        full_problog_program = " ".join(
            context_problog + problog_rules + ["query(action(_))."]
        )
        query_program = (
            get_evaluatable()
            .create_from(PrologString(full_problog_program))
            .evaluate()
        )
        query_program_str_dict = {str(k): v for k, v in query_program.items()}

        action_probs = np.ndarray((num_actions,))
        for i in range(num_actions):
            action_probs[i] = query_program_str_dict[f"action({i})"]

        if use_argmax:
            action = np.argmax(action_probs)
        else:
            # Sample action
            action = np.random.choice(num_actions, p=action_probs)
        action = int(action)
        trace.append((obs, action, action_probs))

        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward  # type: ignore
        num_frames += 1

    return {
        "episode_reward": episode_reward,
        "trace": trace,
        "num_frames": num_frames,
    }


def prolog_inference_gen_action_dist_for_all_states(
    all_states_context_problog: list[list[str]],
    problog_rules: list[str],
    num_actions: int,
) -> np.ndarray:
    """
    Generate the action distribution for all states in the environment, using
    the given problog rules and `all_states_context_problog`. The
    `all_states_context_problog` should be a list of lists of strings, where
    each inner list represents the context of a state in problog format. Return
    a 2D numpy array of shape (len(all_states_context_problog), num_actions),
    where each row represents the action distribution for a state.
    """
    problog_act_dist = np.zeros((len(all_states_context_problog), num_actions))
    for i, s in enumerate(all_states_context_problog):
        full_problog_program = " ".join(
            s + problog_rules + ["query(action(_))."]
        )
        query_program = (
            get_evaluatable()
            .create_from(PrologString(full_problog_program))
            .evaluate()
        )
        query_program_str_dict = {str(k): v for k, v in query_program.items()}

        for j in range(num_actions):
            problog_act_dist[i, j] = query_program_str_dict[f"action({j})"]

    return problog_act_dist
