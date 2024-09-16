from enum import IntEnum
import logging
from typing import Any, Callable

import clingo
from gymnasium import Env

log = logging.getLogger()

DEFAULT_EVAL_NUM_EPISODES = 10


class ASPRuleEvaluationFailureCode(IntEnum):
    FAIL_AT_RULE_EVAL_NOT_ONE_ANSWER_SET = -1
    FAIL_AT_RULE_EVAL_MISSING_ACTION = -2
    FAIL_AT_RULE_EVAL_MORE_THAN_ONE_ACTION = -3
    FAIL_AT_RULE_EVAL_TRUNCATED = -4


def _single_env_eval(
    env: Env,
    context_encoding_generation: Callable[[Any], list[str]],
    num_actions: int,
    rules: list[str],
    do_logging: bool = False,
) -> ASPRuleEvaluationFailureCode | float:
    obs, _ = env.reset()

    terminated = False
    truncated = False
    reward_sum = 0

    while not terminated and not truncated:
        context = context_encoding_generation(obs)

        ctl = clingo.Control(["--warn=none"])
        show_statements = [f"#show disj_{i}/0." for i in range(num_actions)]
        ctl.add("base", [], " ".join(context + show_statements + rules))
        ctl.ground([("base", [])])
        with ctl.solve(yield_=True) as handle:  # type: ignore
            all_answer_sets = [str(a) for a in handle]

        if len(all_answer_sets) != 1:
            # No model or multiple answer sets, should not happen
            log.info(f"No model or multiple answer sets when evaluating rules.")
            return (
                ASPRuleEvaluationFailureCode.FAIL_AT_RULE_EVAL_NOT_ONE_ANSWER_SET
            )

        if all_answer_sets[0] == "":
            log.info(f"No output action!")
            return ASPRuleEvaluationFailureCode.FAIL_AT_RULE_EVAL_MISSING_ACTION

        output_classes = all_answer_sets[0].split(" ")
        if len(output_classes) == 0:
            log.info(f"No output action!")
            return ASPRuleEvaluationFailureCode.FAIL_AT_RULE_EVAL_MISSING_ACTION
        output_classes_set = set([int(o[5:]) for o in output_classes])

        if len(output_classes_set) != 1:
            log.info(f"Output set: {output_classes_set} not exactly one item!")
            return (
                ASPRuleEvaluationFailureCode.FAIL_AT_RULE_EVAL_MORE_THAN_ONE_ACTION
            )

        action = list(output_classes_set)[0]
        if do_logging:
            log.info(f"Action: {action}")
        obs, reward, terminated, truncated, _ = env.step(action)
        reward_sum += reward  # type: ignore

    if truncated:
        log.info(f"Truncated: {reward_sum}")
        return ASPRuleEvaluationFailureCode.FAIL_AT_RULE_EVAL_TRUNCATED

    if do_logging:
        log.info(f"Reward sum: {reward_sum}")

    return reward_sum


def evaluate_rule_on_env(
    env: Env,
    context_encoding_generation_fn: Callable[[Any], list[str]],
    num_actions: int,
    rules: list[str],
    eval_num_episodes: int = DEFAULT_EVAL_NUM_EPISODES,
    do_logging: bool = False,
) -> ASPRuleEvaluationFailureCode | list[float]:
    all_reward_sum = []
    for _ in range(eval_num_episodes):
        ret = _single_env_eval(
            env, context_encoding_generation_fn, num_actions, rules, do_logging
        )
        if isinstance(ret, ASPRuleEvaluationFailureCode):
            return ret
        else:
            all_reward_sum.append(ret)
    return all_reward_sum
