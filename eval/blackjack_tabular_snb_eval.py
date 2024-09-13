# This script evaluates the tabular agents described in Sutton and Barto's book
# in the Blackjack environment.
import json
import logging
from pathlib import Path
import random
import sys
import traceback


import gymnasium as gym
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


from blackjack_common import get_target_policy
from eval.blackjack_tabular_multirun_eval import result_analysis
from utils import post_to_discord_webhook

DEFAULT_GEN_SEED = 3
BLACKJACK_SINGLE_ENV_NUM_EPISODES = 1000000


log = logging.getLogger()


def result_analysis(res_list: list[float]) -> dict[str, int | float]:
    res_array = np.array(res_list)

    # Avg. return
    avg_return = float(np.mean(res_array))
    # Std. return
    std_return = float(np.std(res_array))
    # Ste. return
    ste_return = std_return / np.sqrt(len(res_list))

    log.info(f"Avg. return: {avg_return:.3f}")
    log.info(f"Std. return: {std_return:.3f}")
    log.info(f"Ste. return: {ste_return:.3f}")

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
        "ste_return": ste_return,
        "num_wins": num_wins,
        "win_rate": win_rate,
        "num_losses": num_losses,
        "loss_rate": loss_rate,
        "num_draws": num_draws,
        "draw_rate": draw_rate,
        "num_episodes": num_episodes,
    }


def eval(cfg: DictConfig) -> dict[str, int | float]:
    """
    Evaluate the tabular agent over a number of episodes across the environment.
    Return the results in a dictionary.
    """
    target_policy_csv_path = Path(cfg["eval"]["target_policy_csv_path"])
    assert (
        target_policy_csv_path.exists()
    ), f"Path {target_policy_csv_path} does not exist."
    target_policy = get_target_policy(target_policy_csv_path)

    log.info(f"Evaluating Sutton and Barto's tabular agent:")

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

    ret_dict = result_analysis(res_list)
    with open("snb_eval_results.json", "w") as f:
        json.dump(ret_dict, f, indent=4)

    return ret_dict


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
        ret_dict = eval(cfg)

        if use_discord_webhook:
            msg_body = "Success!\n"
            msg_body += "Sutton and Barto's tabular agent evaluation results:\n"
            msg_body += f"Avg return:\t{ret_dict['avg_return']}\n"
            msg_body += f"Std return:\t{ret_dict['std_return']}\n"
            msg_body += f"Ste return:\t{ret_dict['ste_return']}\n"
            msg_body += f"Win rate:\t{ret_dict['win_rate']}\n"
            msg_body += f"Loss rate:\t{ret_dict['loss_rate']}\n"
            msg_body += f"Draw rate:\t{ret_dict['draw_rate']}\n"
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
    torch.set_warn_always(False)

    run_eval()
