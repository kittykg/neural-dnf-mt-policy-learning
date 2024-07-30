import logging
from pathlib import Path
import random
import time
import traceback
from typing import Any

import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo
import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn, Tensor, optim
from torch.utils.tensorboard import SummaryWriter  # type: ignore
import wandb


from neural_dnf.neural_dnf import NeuralDNFMutexTanh
from neural_dnf.utils import DeltaDelayedExponentialDecayScheduler

from blackjack_common import *
from common import DiversityScoreTracker
from utils import post_to_discord_webhook

A2C_MODEL_DIR = Path(__file__).parent / "blackjack_a2c_storage/"
if not A2C_MODEL_DIR.exists() or not A2C_MODEL_DIR.is_dir():
    A2C_MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TARGET_STATE = (16, 5, 1)
log = logging.getLogger()


log = logging.getLogger()


def train_ac(
    training_cfg: DictConfig,
    full_experiment_name: str,
    use_wandb: bool,
    writer: SummaryWriter,
    save_model: bool = True,
) -> tuple[dict[str, Any], Path, BlackjackBaseAgent]:
    use_ndnf = "ndnf" in full_experiment_name
    use_decode_obs = training_cfg["use_decode_obs"]

    use_cuda = torch.cuda.is_available() and training_cfg["use_cuda"]

    use_mps = (
        torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
        and training_cfg.get("use_mps", False)
    )

    if use_mps:
        device = torch.device("mps")
    else:
        device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )

    print(f"Training on device: {device}")

    # Env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(i, i, False) for i in range(training_cfg["num_envs"])]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    # Load training status
    agent = construct_model(
        num_latent=training_cfg["model_latent_size"],
        use_ndnf=use_ndnf,
        use_decode_obs=use_decode_obs,
        use_eo="use_eo" in training_cfg and training_cfg["use_eo"],
        use_mt="use_mt" in training_cfg and training_cfg["use_mt"],
    )
    agent.train()
    agent.to(device)

    optimizer = optim.Adam(
        agent.parameters(), lr=training_cfg["learning_rate"], eps=1e-5
    )

    # Load delta delayed exponential decay scheduler if the model uses a neural
    # DNF based actor
    if isinstance(agent, BlackjackNDNFBasedAgent):
        dds_cfg = training_cfg["dds"]
        dds = DeltaDelayedExponentialDecayScheduler(
            initial_delta=dds_cfg["initial_delta"],
            delta_decay_delay=dds_cfg["delta_decay_delay"],
            delta_decay_steps=dds_cfg["delta_decay_steps"],
            delta_decay_rate=dds_cfg["delta_decay_rate"],
            target_module_type=agent.actor.__class__.__name__,
        )

    # A2C parameters
    num_frames_per_proc = training_cfg["num_frames_per_proc"]
    discount = training_cfg["discount"]
    gae_lambda = training_cfg["gae_lambda"]
    entropy_coef = training_cfg["entropy_coef"]
    value_loss_coef = training_cfg["value_loss_coef"]
    max_grad_norm = training_cfg["max_grad_norm"]
    num_procs = training_cfg["num_envs"]
    num_frames = num_frames_per_proc * num_procs
    save_interval = training_cfg["save_interval"]

    # A2c storage
    num_inputs = agent.num_inputs
    obs_shape = (num_frames_per_proc, num_procs, num_inputs)
    obs = torch.zeros(obs_shape).to(device)

    actions = torch.zeros((num_frames_per_proc, num_procs)).to(device)
    log_probs = torch.zeros((num_frames_per_proc, num_procs)).to(device)
    rewards = torch.zeros((num_frames_per_proc, num_procs)).to(device)
    dones = torch.zeros((num_frames_per_proc, num_procs)).to(device)
    values = torch.zeros((num_frames_per_proc, num_procs)).to(device)

    # Training
    global_step = 0
    delta_one_count = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_obs_dict = blackjack_env_preprocess_obss(next_obs, use_ndnf, device)
    next_done = torch.zeros(num_procs, device=device)

    aux_loss_enable_log = False
    last_episodic_return = None

    while global_step < int(training_cfg["total_num_frames"]):
        # Collect experiences
        for i in range(num_frames_per_proc):
            global_step += num_procs
            obs[i] = next_obs_dict[agent.input_key]
            dones[i] = next_done

            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(
                    next_obs_dict
                )

            values[i] = value.flatten()
            actions[i] = action
            log_probs[i] = log_prob

            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[i] = torch.tensor(reward).float().to(device).view(-1)
            next_obs_dict = blackjack_env_preprocess_obss(
                next_obs, use_ndnf, device
            )
            next_done = torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        if (
                            iteration % training_cfg["log_interval"] == 0
                            and step == num_steps - 1
                        ):
                            print(
                                f"global_step={global_step}, "
                                f"episodic_return={info['episode']['r']}"
                            )
                        writer.add_scalar(
                            "charts/episodic_return",
                            info["episode"]["r"],
                            global_step,
                        )
                        last_episodic_return = info["episode"]["r"]
                        writer.add_scalar(
                            "charts/episodic_length",
                            info["episode"]["l"],
                            global_step,
                        )

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs_dict).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            last_gae_lam = 0
            for t in reversed(range(num_frames_per_proc)):
                if t == num_frames_per_proc - 1:
                    next_non_terminal = 1.0 - next_done
                    next_vals = next_value
                else:
                    next_non_terminal = 1.0 - dones[t + 1]
                    next_vals = values[t + 1]
                delta = (
                    rewards[t]
                    + training_cfg["gamma"] * next_vals * next_non_terminal
                    - values[t]
                )
                advantages[t] = last_gae_lam = (
                    delta
                    + training_cfg["gamma"]
                    * training_cfg["gae_lambda"]
                    * next_non_terminal
                    * last_gae_lam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1, num_inputs))
        b_log_probs = log_probs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)  # type: ignore
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        for _ in range(training_cfg["update_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, new_log_prob, entropy, new_val = agent.get_action_and_value(
                    {agent.input_key: b_obs[mb_inds]}, b_actions.long()[mb_inds]
                )
                log_ratio = new_log_prob - b_log_probs[mb_inds]
                ratio = log_ratio.exp()

                mb_advantages = b_advantages[mb_inds]

                # Policy loss
                # TODO: MODIFY IT TO MATCH WITH A2C
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio,
                    1 - training_cfg["clip_coef"],
                    1 + training_cfg["clip_coef"],
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                new_val = new_val.view(-1)
                v_loss = 0.5 * ((new_val - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    - training_cfg["ent_coef"] * entropy_loss
                    + v_loss * training_cfg["vf_coef"]
                )

                optimizer.zero_grad(set_to_none=True)

                # Auxiliary loss for NDNF based agents
                if isinstance(agent, BlackjackNDNFBasedAgent):
                    if agent.actor.get_delta_val()[
                        0
                    ] == 1 and delta_one_count > training_cfg["aux_loss"].get(
                        "delta_one_delay", 3
                    ):
                        aux_loss_dict = agent.get_aux_loss(next_obs_dict)

                        l_disj_l1_mod_lambda = training_cfg["aux_loss"][
                            "dis_l1_mod_lambda"
                        ]
                        l_disj_l1_mod = aux_loss_dict["l_disj_l1_mod"]
                        loss += l_disj_l1_mod_lambda * l_disj_l1_mod

                        l_tanh_conj_lambda = training_cfg["aux_loss"][
                            "tanh_conj_lambda"
                        ]
                        l_tanh_conj = aux_loss_dict["l_tanh_conj"]
                        loss += l_tanh_conj_lambda * l_tanh_conj

                        if isinstance(agent, BlackjackNDNFMutexTanhAgent):
                            l_mt_ce2_lambda = training_cfg["aux_loss"][
                                "mt_ce2_lambda"
                            ]
                            l_mt_ce2 = aux_loss_dict["l_mt_ce2"]
                            loss += l_mt_ce2_lambda * l_mt_ce2
                    elif agent.actor.get_delta_val()[0] == 1:
                        delta_one_count += 1

                loss.backward()
                nn.utils.clip_grad_norm_(  # type: ignore
                    [p for p in agent.parameters() if p.requires_grad],
                    training_cfg["max_grad_norm"],
                )
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = (
            np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y  # type: ignore
        )

        if isinstance(agent, BlackjackNDNFBasedAgent):
            delta_dict = dds.step(agent.actor)
            new_delta = delta_dict["new_delta_vals"][0]
            old_delta = delta_dict["old_delta_vals"][0]

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar(
            "losses/explained_variance", explained_var, global_step
        )
        if isinstance(agent, BlackjackNDNFBasedAgent):
            if agent.actor.get_delta_val()[
                0
            ] == 1 and delta_one_count > training_cfg["aux_loss"].get(
                "delta_one_delay", 3
            ):
                writer.add_scalar(
                    "losses/l_disj_l1_mod", l_disj_l1_mod.item(), global_step
                )
                writer.add_scalar(
                    "losses/l_tanh_conj", l_tanh_conj.item(), global_step
                )

                if isinstance(agent, BlackjackNDNFMutexTanhAgent):
                    writer.add_scalar(
                        "losses/l_mt_ce2", l_mt_ce2.item(), global_step
                    )

        if isinstance(agent, BlackjackNDNFBasedAgent):
            writer.add_scalar("charts/delta", old_delta, global_step)  # type: ignore
            if new_delta != old_delta:  # type: ignore
                print(
                    f"i={iteration}\t"
                    f"old delta={old_delta:.3f} new delta={new_delta:.3f}\t"
                    f"last episodic_return={last_episodic_return}"
                )  # type: ignore

        writer.add_scalar(
            f"charts/action_hit_prob_at_{DEFAULT_TARGET_STATE}",
            track_target_state_action_distribution(agent, device),
            global_step,
        )
        writer.add_scalar(
            "charts/SPS",
            int(global_step / (time.time() - start_time)),
            global_step,
        )

    envs.close()
    writer.close()

    eval_log = None
    if "plot_policy" in training_cfg and training_cfg["plot_policy"]:
        assert "target_policy_csv_path" in training_cfg
        assert training_cfg["target_policy_csv_path"] is not None

        if isinstance(agent, BlackjackNDNFEOAgent):
            eval_agent = agent.to_ndnf_agent()
        else:
            eval_agent = agent
        eval_agent.eval()

        if isinstance(eval_agent, BlackjackNDNFBasedAgent):
            eval_log = ndnf_based_agent_cmp_target_csv(
                training_cfg["target_policy_csv_path"], eval_agent, device
            )
        else:
            eval_log = mlp_agent_cmp_target_csv(
                training_cfg["target_policy_csv_path"],
                eval_agent,  # type: ignore
                device,
                training_cfg["normalise_obs"],
            )

        log.info(eval_log)
        if use_wandb:
            mod_logs = {}
            for k, v in eval_log.items():
                if isinstance(v, bool):
                    mod_logs[f"ndnf_based_agent/{k}"] = int(v)
                elif isinstance(v, list):
                    continue
                else:
                    mod_logs[f"ndnf_based_agent/{k}"] = v
            wandb.log(mod_logs)

        plot_policy_grid_after_train(
            training_cfg["target_policy_csv_path"],
            eval_agent,
            device,
            full_experiment_name,
            use_wandb,
        )

    if save_model:
        model_dir = PPO_MODEL_DIR / full_experiment_name
        if not model_dir.exists() or not model_dir.is_dir():
            model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pth"
        torch.save(agent.state_dict(), model_path)
        return model_path, agent, eval_log

    return None, agent, eval_log


def after_train_eval(
    model: BlackjackEnvNDNFMTAgent,
    device: torch.device,
    use_wandb: bool,
    txt_logger,
):
    model.eval()
    env = ParallelBlackJackEnv([construct_train_env() for _ in range(8)])
    simulate = lambda action_fn: simulate_fn(env, action_fn, num_episodes=12)

    def _simulate_with_log(action_fn, model_name: str):
        logs = simulate(action_fn)

        num_frames = sum(logs["num_frames_per_episode"])
        return_per_episode = synthesize(logs["return_per_episode"])
        num_frames_per_episode = synthesize(logs["num_frames_per_episode"])

        txt_logger.info(
            "{}\tF {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}".format(
                model_name,
                num_frames,
                *return_per_episode.values(),
                *num_frames_per_episode.values(),
            )
        )

        txt_logger.info(f"Mutual exclusivity: {logs['mutual_exclusivity']}")
        txt_logger.info(f"Missing actions: {logs['missing_actions']}")

        header = []
        data = []
        header += [
            "post_train_eval_rreturn_" + key
            for key in return_per_episode.keys()
        ]
        data += return_per_episode.values()
        header += [
            "post_train_eval_num_frames_" + key
            for key in num_frames_per_episode.keys()
        ]
        data += num_frames_per_episode.values()
        header += [
            "post_train_eval_mutual_exclusivity",
            "post_train_eval_missing_actions",
        ]
        data += [logs["mutual_exclusivity"], logs["missing_actions"]]

        if use_wandb:
            log_dict = dict(zip(header, data))
            wandb.log(log_dict)

    def get_ndnf_mt_action(
        model: BlackjackEnvNDNFMTAgent, discretise_embedding: bool, obs
    ):
        # Use normal tanh interpretation
        with torch.no_grad():
            actions = model.get_actions(
                preprocessed_obs=blackjack_env_preprocess_obss(obs, device),  # type: ignore
                use_argmax=True,
            )
        return actions

    base_action_fn = get_ndnf_mt_action
    _ndnf_mt_action_fn = lambda obs: base_action_fn(model, False, obs)
    _ndnf_mt_dis_action_fn = lambda obs: base_action_fn(model, True, obs)
    txt_logger.info("Post train eval:")
    _simulate_with_log(_ndnf_mt_action_fn, "NDNF MT")
    _simulate_with_log(_ndnf_mt_dis_action_fn, "NDNF MT Dis")

    env.close()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    training_cfg = cfg["training"]
    seed = training_cfg["seed"]
    full_experiment_name = training_cfg["experiment_name"] + f"_{seed}"

    # Set random seed
    torch.manual_seed(seed)
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
        )

    torch.autograd.set_detect_anomaly(True)  # type: ignore

    use_discord_webhook = cfg["webhook"]["use_discord_webhook"]
    msg_body = None
    errored = False

    try:
        _, status_path, _ = train_ac(
            training_cfg, full_experiment_name, use_wandb
        )
        if use_wandb:
            wandb.save(glob_str=str(status_path.absolute()))

        if use_discord_webhook:
            msg_body = "Success!"
    except BaseException as e:
        if use_discord_webhook:
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
            )
        if use_wandb:
            wandb.finish()
        if not errored:
            path = Path(HydraConfig.get().run.dir)
            path.rename(path.absolute().parent / run_dir_name)


if __name__ == "__main__":
    import multiprocessing as mp

    if mp.get_start_method() != "fork":
        mp.set_start_method("fork", force=True)

    run_experiment()
