# This script evaluates the tabular agents trained on the Blackjack environment.
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
import torch


file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from blackjack_common import *
from utils import post_to_discord_webhook

DEFAULT_GEN_SEED = 3
BLACKJACK_SINGLE_ENV_NUM_EPISODES = 500


log = logging.getLogger()


def result_analysis(res_list: list[float]) -> dict[str, int | float]:
    res_array = np.array(res_list)

    # Avg. return
    avg_return = float(np.mean(res_array))
    # Std. return
    std_return = float(np.std(res_array))

    log.info(f"Avg. return: {avg_return:.3f}")
    log.info(f"Std. return: {std_return:.3f}")

    num_episodes = len(res_list)
    num_wins = int(np.sum(np.array(res_list) == 1))
    win_rate = num_wins / num_episodes
    num_losses = int(np.sum(np.array(res_list) == -1))
    loss_rate = num_losses / num_episodes
    num_draws = int(np.sum(np.array(res_list) == 0))
    draw_rate = num_draws / num_episodes

    log.info(f"Average reward: {np.mean(res_list)}")
    log.info(f"Number of wins: {num_wins}\tPercentage: {win_rate}")
    log.info(f"Number of losses: {num_losses}\tPercentage: {loss_rate}")
    log.info(f"Number of draws: {num_draws}\tPercentage: {draw_rate}")

    np.set_printoptions(formatter={"float": lambda x: "{:.3f}".format(x)})

    return {
        "avg_return": avg_return,
        "std_return": std_return,
        "num_wins": num_wins,
        "win_rate": win_rate,
        "num_losses": num_losses,
        "loss_rate": loss_rate,
        "num_draws": num_draws,
        "draw_rate": draw_rate,
        "num_episodes": num_episodes,
    }


def single_eval(
    experiment_name: str, target_policy_csv_path: Path
) -> dict[str, int | float]:
    """
    Evaluate the tabular agent over a number of episodes across the environment.
    Return the results in a dictionary.
    """
    target_policy = get_target_policy(target_policy_csv_path)

    log.info(f"Evaluating tabular agent: {experiment_name}")

    res_list = []

    env = gym.make("Blackjack-v1", render_mode="rgb_array")

    for _ in range(BLACKJACK_SINGLE_ENV_NUM_EPISODES):
        obs, _ = env.reset()
        terminated, truncated = False, False
        reward_sum = 0
        while not terminated and not truncated:
            action = target_policy[obs]
            obs, reward, terminated, truncated, _ = env.step(action)
            reward_sum += reward  # type: ignore
        res_list.append(reward_sum)

    log.info(f"Results of {experiment_name}:")
    ret_dict = result_analysis(res_list)

    return ret_dict


def multirun_eval(cfg: DictConfig) -> dict[str, Any]:
    eval_cfg = cfg["eval"]
    blackjack_tab_storage_dir = Path(eval_cfg["blackjack_tab_storage_dir"])
    assert (
        blackjack_tab_storage_dir.exists()
        and blackjack_tab_storage_dir.is_dir()
    ), f"Directory {blackjack_tab_storage_dir} does not exist."

    skip_experiments = eval_cfg.get("skip_experiments", [])

    ret_dicts = []

    # Iterator over all the subdirectories
    for d in blackjack_tab_storage_dir.iterdir():
        experiment_name = d.stem
        if experiment_name in skip_experiments:
            log.info(f"Skipping experiment {experiment_name}")
            continue

        policy_name = "_".join(experiment_name.lower().split("-"))
        target_policy_path = d / f"{policy_name}.csv"
        if not target_policy_path.exists():
            log.error(f"Path {target_policy_path} does not exist.")
            continue

        r = single_eval(experiment_name, target_policy_path)
        ret_dicts.append(r)

        # Save the return dict into a json
        with open(d / "eval_results.json", "w") as f:
            json.dump(r, f)

        log.info("=====================================")

    # Overall results
    overall_avg = np.mean([ret_dict["avg_return"] for ret_dict in ret_dicts])
    overall_std = np.std([ret_dict["avg_return"] for ret_dict in ret_dicts])
    overall_win_rate = np.mean([ret_dict["win_rate"] for ret_dict in ret_dicts])
    overall_loss_rate = np.mean(
        [ret_dict["loss_rate"] for ret_dict in ret_dicts]
    )
    overall_draw_rate = np.mean(
        [ret_dict["draw_rate"] for ret_dict in ret_dicts]
    )

    log.info(f"Results of {len(ret_dicts)} tabular agents:")
    log.info(f"Overall avg return: {overall_avg}")
    log.info(f"Overall std return: {overall_std}")
    log.info(f"Overall win rate: {overall_win_rate}")
    log.info(f"Overall loss rate: {overall_loss_rate}")
    log.info(f"Overall draw rate: {overall_draw_rate}")

    return {
        "overall_avg": overall_avg,
        "overall_std": overall_std,
        "overall_win_rate": overall_win_rate,
        "overall_loss_rate": overall_loss_rate,
        "overall_draw_rate": overall_draw_rate,
    }


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run_eval(cfg: DictConfig) -> None:
    # Set random seed
    torch.manual_seed(DEFAULT_GEN_SEED)
    np.random.seed(DEFAULT_GEN_SEED)
    random.seed(DEFAULT_GEN_SEED)

    torch.autograd.set_detect_anomaly(True)  # type: ignore

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
            msg_body += f"Overall win rate:\t{ret_dict['overall_win_rate']}\n"
            msg_body += f"Overall loss rate:\t{ret_dict['overall_loss_rate']}\n"
            msg_body += f"Overall draw rate:\t{ret_dict['overall_draw_rate']}\n"
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
