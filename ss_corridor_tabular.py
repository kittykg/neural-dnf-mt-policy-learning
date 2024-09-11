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

from corridor_grid.envs.base_ss_corridor import BaseSpecialStateCorridorEnv
from ss_corridor_ppo import construct_single_environment
from tabular_common import TabularQAgent, get_moving_average_for_plot
from utils import post_to_discord_webhook

log = logging.getLogger()

# This map is used for mapping a wall status observation to a single index for
# the Q table.
# There should be only 3 combinations of wall status in a SSCorridorEnv, since
# there is no situation where both walls are present on the left and right.
SSCorridorWallStatusMap: dict[tuple[int, int], int] = {
    (0, 0): 0,
    (1, 0): 1,
    (0, 1): 2,
}
SSCorridorWallStatusReverseMap: dict[int, tuple[int, int]] = {
    0: (0, 0),
    1: (1, 0),
    2: (0, 1),
}
N_ACTIONS = 2


class SSCTabularQAgent(TabularQAgent):
    n_states: int
    n_actions: int
    use_state_no_as_obs: bool
    q_table: npt.NDArray[np.float64]

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        use_state_no_as_obs: bool,
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
        self.use_state_no_as_obs = use_state_no_as_obs
        self.q_table = np.zeros((n_states, n_actions))

    def _obs_processing(self, obs: dict) -> int:
        return (
            SSCorridorWallStatusMap[tuple(obs["wall_status"])]  # type: ignore
            if not self.use_state_no_as_obs
            else obs["agent_location"]
        )


def train(
    env: BaseSpecialStateCorridorEnv,
    agent: SSCTabularQAgent,
    num_episodes: int,
    use_state_no_as_obs: bool,
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

        episode_durations.append(episode_duration)
        episode_rewards.append(total_reward)

        if i % logging_freq == 0:
            log.info(
                f"Episode {i}\t"
                f"Epsilon threshold: {eps_threshold:.2f}\t"
                f"Reward: {total_reward}\t"
                f"Duration: {episode_duration}"
            )

            if use_wandb:
                # Plot the moving average of the rewards, durations and td errors
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

                # Log the Q table
                for i in range(agent.n_states):
                    for j in range(agent.n_actions):
                        k = (
                            f"S{i}_A{j}"
                            if use_state_no_as_obs
                            else f"S{SSCorridorWallStatusReverseMap[i]}_A{j}"
                        )
                        v = agent.q_table[i, j]
                        log_dict[k] = v
                wandb.log(log_dict)

    log.info("Training complete")

    log.info("Table:")
    log.info("\t" + "\t".join([str(i) for i in range(agent.n_actions)]))
    for i in range(agent.n_states):
        info_str = (
            f"S{i}\t"
            if use_state_no_as_obs
            else f"{SSCorridorWallStatusReverseMap[i]}\t"
        )
        info_str += "\t".join(
            [f"{agent.q_table[i, j]:.2f}" for j in range(agent.n_actions)]
        )
        log.info(f"{info_str}")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    training_cfg = cfg["training"]
    seed = training_cfg["seed"]
    if seed is None:
        seed = random.randint(0, 10000)

    # Expect the experiment name to be in the format of
    # [CORRIDOR SHORT CODE]_tab..._..._..._...
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

    env = construct_single_environment(training_cfg, render_mode="ansi")
    use_state_no_as_obs = training_cfg["use_state_no_as_obs"]
    n_states = (
        env.corridor_length
        if use_state_no_as_obs
        else len(SSCorridorWallStatusMap)
    )
    agent = SSCTabularQAgent(
        n_states=n_states,
        n_actions=int(env.action_space.n),
        use_state_no_as_obs=use_state_no_as_obs,
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
            use_state_no_as_obs,
            use_wandb,
            logging_freq=training_cfg["logging_freq"],
        )

        table_name = f"{full_experiment_name}.npy"
        df = pd.DataFrame(agent.q_table)
        df.to_csv(table_name)

        if use_wandb:
            model_artifact = wandb.Artifact(
                table_name,
                type="ndarray",
                description=f"{full_experiment_name} Q-value Table",
                metadata=dict(wandb.config),
            )
            model_artifact.add_file(table_name)
            wandb.save(table_name)
            run.log_artifact(model_artifact)  # type: ignore

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
