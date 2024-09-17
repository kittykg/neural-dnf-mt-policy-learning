# This script evaluates the tabular agents trained on the SpecialStateCorridor
# environment.
import json
import logging
from pathlib import Path
import random
import sys
import traceback
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig
import pandas as pd


file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass


from ss_corridor_ppo import construct_single_environment
from ss_corridor_tabular import SSCorridorWallStatusMap, N_ACTIONS
from utils import post_to_discord_webhook

DEFAULT_GEN_SEED = 3
SSC_SINGLE_ENV_NUM_EPISODES = 1000000
BASE_STORAGE_DIR = root / "results"


log = logging.getLogger()


def get_action_from_q_table(
    q_table: np.ndarray,
    use_state_no_as_obs: bool,
    obs: dict[str, Any],
    use_argmax: bool,
    epsilon: float,
) -> int:
    i = (
        SSCorridorWallStatusMap[tuple(obs["wall_status"])]
        if not use_state_no_as_obs
        else obs["agent_location"]
    )

    if use_argmax:
        return int(np.argmax(q_table[i]))
    else:
        if np.random.rand() < epsilon:
            return np.random.randint(N_ACTIONS)
        return int(np.argmax(q_table[i]))


def result_analysis(res_list: list[float]) -> dict[str, int | float]:
    res_array = np.array(res_list)

    # Avg. return
    avg_return = float(np.mean(res_array))
    # Std. return
    std_return = float(np.std(res_array))
    # Std. error
    std_error = std_return / np.sqrt(len(res_list))

    log.info(f"Avg. return: {avg_return:.3f}")
    log.info(f"Std. return: {std_return:.3f}")
    log.info(f"Std. error: {std_error:.3f}")

    return {
        "avg_return": avg_return,
        "std_return": std_return,
        "std_error": std_error,
    }


def single_eval(
    eval_cfg: DictConfig,
    seed: int,
) -> dict[str, int | float]:
    """
    Evaluate the tabular agent over a number of episodes across the environment.
    Return the results in a dictionary.
    """
    experiment_name = eval_cfg["experiment_name"]
    use_state_no_as_obs = "sn" in experiment_name
    use_argmax = eval_cfg.get("use_argmax", True)
    epsilon = eval_cfg.get("epsilon", 0.1)

    name_list = experiment_name.split("_")
    # Capitalise the first 3 words in name_list
    env_name = name_list[0]
    name_list = [name.upper() for name in name_list[:3]] + name_list[3:]
    dir_name = "-".join(name_list)

    base_dir = (
        BASE_STORAGE_DIR
        / {"sc": "SC-TAB", "lc5": "LC5-TAB", "lc11": "LC11-TAB"}[env_name]
    )

    if "sn" in experiment_name:
        base_dir = base_dir / "SN" / dir_name / f"{dir_name}-{seed}"
    else:
        base_dir = base_dir / "WS" / dir_name / f"{dir_name}-{seed}"

    target_q_table_csv_path = base_dir / f"{experiment_name}_{seed}.csv"

    with open(target_q_table_csv_path, "r") as f:
        # read the data as pandas dataframe
        df = pd.read_csv(f, index_col=0)

    target_q_table = df.to_numpy()

    log.info(f"Evaluating tabular agent: {experiment_name}_{seed}")

    res_list = []

    env = construct_single_environment(eval_cfg)

    for _ in range(SSC_SINGLE_ENV_NUM_EPISODES):
        obs, _ = env.reset()
        terminated, truncated = False, False
        reward_sum = 0
        while not terminated and not truncated:
            action = get_action_from_q_table(
                target_q_table, use_state_no_as_obs, obs, use_argmax, epsilon
            )
            obs, reward, terminated, truncated, _ = env.step(action)
            reward_sum += reward  # type: ignore
        res_list.append(reward_sum)

    log.info(f"Results of {experiment_name}:")
    ret_dict = result_analysis(res_list)

    # Save the return dict into a json
    with open(base_dir / "eval_results.json", "w") as f:
        json.dump(ret_dict, f)

    return ret_dict


def multirun_eval(cfg: DictConfig) -> dict[str, Any]:
    eval_cfg = cfg["eval"]
    multirun_seeds = eval_cfg["multirun_seeds"]
    ret_dicts = []

    # Iterator over all the subdirectories
    for seed in multirun_seeds:
        r = single_eval(eval_cfg, int(seed))
        ret_dicts.append(r)

        log.info("=====================================")

    # Overall results
    overall_avg = np.mean([ret_dict["avg_return"] for ret_dict in ret_dicts])
    overall_std = np.std([ret_dict["avg_return"] for ret_dict in ret_dicts])
    overall_ste = overall_std / np.sqrt(len(ret_dicts))

    log.info(f"Results of {len(ret_dicts)} tabular agents:")
    log.info(f"Overall avg return: {overall_avg}")
    log.info(f"Overall std return: {overall_std}")
    log.info(f"Overall std error: {overall_ste}")

    return {
        "overall_avg": overall_avg,
        "overall_std": overall_std,
        "overall_ste": overall_ste,
    }


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run_eval(cfg: DictConfig) -> None:
    # Set random seed
    np.random.seed(DEFAULT_GEN_SEED)
    random.seed(DEFAULT_GEN_SEED)

    use_discord_webhook = cfg["webhook"]["use_discord_webhook"]
    msg_body = None
    keyboard_interrupt = None
    errored = False

    try:
        ret_dict = multirun_eval(cfg)

        if use_discord_webhook:
            msg_body = "Success!\n"
            msg_body += f"Overall avg return:\t{ret_dict['overall_avg']}\n"
            msg_body += f"Overall std return:\t{ret_dict['overall_std']}\n"
            msg_body += f"Overall std error:\t{ret_dict['overall_ste']}"
    except BaseException as e:
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
                experiment_name="Blackjack Tabular Multirun Eval",
                message_body=msg_body,
                errored=errored,
                keyboard_interrupt=keyboard_interrupt,
            )


if __name__ == "__main__":
    run_eval()
