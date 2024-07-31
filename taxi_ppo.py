import logging
import random
from pathlib import Path
import time
import traceback
from typing import Any

import gymnasium as gym
import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # type: ignore
import wandb

from neural_dnf.utils import DeltaDelayedExponentialDecayScheduler

from common import synthesize
from taxi_common import *
from utils import post_to_discord_webhook

PPO_MODEL_DIR = Path(__file__).parent / "taxi_ppo_storage/"
if not PPO_MODEL_DIR.exists() or not PPO_MODEL_DIR.is_dir():
    PPO_MODEL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_NUM_RUNS = 10

log = logging.getLogger()


def eval_model_on_environment(
    model: TaxiEnvPPOBaseAgent,
    device: torch.device = torch.device("cpu"),
    use_argmax: bool = True,
    eval_num_runs: int = EVAL_NUM_RUNS,
) -> dict[str, Any]:
    model.to(device)
    use_ndnf = isinstance(model, TaxiEnvPPONDNFBasedAgent)

    logs: dict[str, Any] = {
        "num_frames_per_episode": [],
        "return_per_episode": [],
    }
    if use_ndnf and use_argmax:
        logs["mutual_exclusivity"] = True
        logs["missing_actions"] = False

    envs = gym.vector.SyncVectorEnv(
        [make_env(i, i, False) for i in range(eval_num_runs)],
    )

    next_obs, _ = envs.reset()
    next_obs_dict = taxi_env_preprocess_obs(next_obs, use_ndnf, device)

    log_done_counter = 0
    log_episode_return = torch.zeros(eval_num_runs, device=device)
    log_episode_num_frames = torch.zeros(eval_num_runs, device=device)

    while log_done_counter < eval_num_runs:
        if use_ndnf:
            # For NDNF based model, the get_actions() returns a tuple of
            # actions and tanh interpretation. We check the if the tanh
            # interpretation is greater than 0.
            with torch.no_grad():
                actions = model.get_actions(
                    preprocessed_obs=next_obs_dict, use_argmax=use_argmax
                )
            if use_argmax:
                tanh_action = np.count_nonzero(actions[1] > 0, axis=1)
                if np.any(tanh_action > 1):
                    logs["mutual_exclusivity"] = False
                if np.any(tanh_action == 0):
                    logs["missing_actions"] = True
            actions = actions[0]
        else:
            # MLP based model
            with torch.no_grad():
                actions = model.get_actions(
                    preprocessed_obs=next_obs_dict, use_argmax=use_argmax
                )

        next_obs, reward, terminations, truncations, _ = envs.step(actions)
        next_obs_dict = taxi_env_preprocess_obs(next_obs, use_ndnf, device)
        next_done = np.logical_or(terminations, truncations)

        log_episode_return += torch.tensor(
            reward, device=device, dtype=torch.float
        )
        log_episode_num_frames += torch.ones(eval_num_runs, device=device)

        for i, done in enumerate(next_done):
            if done:
                log_done_counter += 1
                logs["return_per_episode"].append(log_episode_return[i].item())
                logs["num_frames_per_episode"].append(
                    log_episode_num_frames[i].item()
                )

        mask = 1 - torch.tensor(next_done, device=device, dtype=torch.float)
        log_episode_return *= mask
        log_episode_num_frames *= mask

    envs.close()

    return logs


def eval_model_on_all_possible_states():
    # TODO: Implement this function
    pass


def train_ppo(
    training_cfg: DictConfig,
    full_experiment_name: str,
    use_wandb: bool,
    writer: SummaryWriter,
    save_model: bool = True,
) -> tuple[Path | None, TaxiEnvPPOBaseAgent, dict[str, dict]]:
    use_ndnf = "ndnf" in full_experiment_name
    use_decode_obs = training_cfg["use_decode_obs"]
    batch_size = int(training_cfg["num_envs"] * training_cfg["num_steps"])
    minibatch_size = int(batch_size // training_cfg["num_minibatches"])
    num_iterations = int(training_cfg["total_timesteps"] // batch_size)

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
        [make_env(i, i, False) for i in range(training_cfg["num_envs"])],
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    agent = construct_model(
        actor_latent_size=training_cfg["actor_latent_size"],
        use_ndnf=use_ndnf,
        use_decode_obs=use_decode_obs,
        use_eo="use_eo" in training_cfg and training_cfg["use_eo"],
        use_mt="use_mt" in training_cfg and training_cfg["use_mt"],
        share_layer_with_critic=training_cfg.get(
            "share_layer_with_critic", False
        ),
        critic_latent_1=training_cfg.get("critic_latent_1", 256),
        critic_latent_2=training_cfg.get("critic_latent_2", 64),
    )
    agent.train()
    agent.to(device)
    log.info(agent)

    optimizer = optim.Adam(
        [
            {
                "params": agent.actor.parameters(),
                "lr": training_cfg["lr_actor"],
            },
            {
                "params": agent.critic.parameters(),
                "lr": training_cfg["lr_critic"],
            },
        ],
        eps=1e-5,
    )

    # ALGO Logic: Storage setup
    num_steps: int = training_cfg["num_steps"]
    num_envs: int = training_cfg["num_envs"]

    num_inputs = agent.num_inputs

    obs_shape = (num_steps, num_envs, num_inputs)
    obs = torch.zeros(obs_shape).to(device)

    action_shape = (num_steps, num_envs) + envs.single_action_space.shape  # type: ignore
    # action_shape: (num_steps, num_envs)
    actions = torch.zeros(action_shape).to(device)

    log_probs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_obs_dict = taxi_env_preprocess_obs(next_obs, use_ndnf, device)
    next_done = torch.zeros(num_envs).to(device)

    if isinstance(agent, TaxiEnvPPONDNFMTAgent):
        dds_cfg = training_cfg["dds"]
        dds = DeltaDelayedExponentialDecayScheduler(
            initial_delta=dds_cfg["initial_delta"],
            delta_decay_delay=dds_cfg["delta_decay_delay"],
            delta_decay_steps=dds_cfg["delta_decay_steps"],
            delta_decay_rate=dds_cfg["delta_decay_rate"],
            target_module_type=agent.actor.__class__.__name__,
        )
        agent.actor.set_delta_val(dds_cfg["initial_delta"])

    for iteration in range(1, num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if training_cfg["anneal_lr"]:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            lr_actor_now = frac * training_cfg["lr_actor"]
            lr_critic_now = frac * training_cfg["lr_critic"]
            optimizer.param_groups[0]["lr"] = lr_actor_now
            optimizer.param_groups[1]["lr"] = lr_critic_now

        for step in range(0, num_steps):
            global_step += num_envs
            obs[step] = next_obs_dict[agent.input_key]
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(
                    next_obs_dict
                )
                values[step] = value.flatten()
            actions[step] = action
            log_probs[step] = log_prob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs_dict = taxi_env_preprocess_obs(next_obs, use_ndnf, device)
            next_done = torch.Tensor(next_done).to(device)

            last_episodic_return = None

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        if (
                            iteration % training_cfg["log_interval"] == 0
                            and step == num_steps - 1
                        ):
                            print(
                                f"global_step={global_step}, episodic_return={info['episode']['r']}"
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
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
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
        b_obs = obs.reshape((-1, num_inputs))  # type: ignore
        b_log_probs = log_probs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)  # type: ignore
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clip_fracs = []
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

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fracs += [
                        ((ratio - 1.0).abs() > training_cfg["clip_coef"])
                        .float()
                        .mean()
                        .item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if training_cfg["norm_adv"]:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio,
                    1 - training_cfg["clip_coef"],
                    1 + training_cfg["clip_coef"],
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                new_val = new_val.view(-1)
                if training_cfg["clip_vloss"]:
                    v_loss_unclipped = (new_val - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        new_val - b_values[mb_inds],
                        -training_cfg["clip_coef"],
                        training_cfg["clip_coef"],
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((new_val - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    - training_cfg["ent_coef"] * entropy_loss
                    + v_loss * training_cfg["vf_coef"]
                )

                optimizer.zero_grad(set_to_none=True)

                if isinstance(agent, TaxiEnvPPONDNFBasedAgent):
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

                    if isinstance(agent, TaxiEnvPPONDNFMTAgent):
                        l_mt_ce2_lambda = training_cfg["aux_loss"][
                            "mt_ce2_lambda"
                        ]
                        l_mt_ce2 = aux_loss_dict["l_mt_ce2"]
                        loss += l_mt_ce2_lambda * l_mt_ce2

                loss.backward()
                nn.utils.clip_grad_norm_(  # type: ignore
                    agent.parameters(), training_cfg["max_grad_norm"]
                )
                optimizer.step()

            if (
                training_cfg["target_kl"] is not None
                and approx_kl > training_cfg["target_kl"]  # type: ignore
            ):
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = (
            np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y  # type: ignore
        )

        if isinstance(agent, TaxiEnvPPONDNFBasedAgent):
            delta_dict = dds.step(agent.actor)
            new_delta = delta_dict["new_delta_vals"][0]
            old_delta = delta_dict["old_delta_vals"][0]

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/lr_actor", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar(
            "charts/lr_critic", optimizer.param_groups[1]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)  # type: ignore
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)  # type: ignore
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)  # type: ignore
        writer.add_scalar(
            "losses/old_approx_kl", old_approx_kl.item(), global_step  # type: ignore
        )
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)  # type: ignore
        writer.add_scalar("losses/clipfrac", np.mean(clip_fracs), global_step)
        writer.add_scalar(
            "losses/explained_variance", explained_var, global_step
        )
        if isinstance(agent, TaxiEnvPPONDNFMTAgent):
            writer.add_scalar(
                "losses/l_disj_l1_mod", l_disj_l1_mod.item(), global_step
            )
            writer.add_scalar(
                "losses/l_tanh_conj", l_tanh_conj.item(), global_step
            )

            if isinstance(agent, TaxiEnvPPONDNFMTAgent):
                writer.add_scalar(
                    "losses/l_mt_ce2", l_mt_ce2.item(), global_step
                )

        if isinstance(agent, TaxiEnvPPONDNFMTAgent):
            writer.add_scalar("charts/delta", old_delta, global_step)  # type: ignore
            if new_delta != old_delta:  # type: ignore
                print(
                    f"i={iteration}\t"
                    f"old delta={old_delta:.3f} new delta={new_delta:.3f}\t"
                    f"last episodic_return={last_episodic_return}"
                )  # type: ignore

        writer.add_scalar(
            "charts/SPS",
            int(global_step / (time.time() - start_time)),
            global_step,
        )

    envs.close()
    writer.close()

    # Evaluate the model
    if isinstance(agent, TaxiEnvPPONDNFEOAgent):
        eval_agent = agent.to_ndnf_agent()
    else:
        eval_agent = agent
    eval_agent.eval()
    argmax_eval_log = eval_model_on_environment(eval_agent, use_argmax=True)
    non_argmax_eval_log = eval_model_on_environment(
        eval_agent, use_argmax=False
    )
    log.info("Argmax evaluation log:")
    log.info(argmax_eval_log)
    log.info("Non-argmax evaluation log:")
    log.info(non_argmax_eval_log)
    if use_wandb:

        def modify_logs(logs: dict[str, Any], suffix: str) -> dict[str, Any]:
            new_logs = {}
            for k, v in logs.items():
                if isinstance(v, bool):
                    new_logs[f"eval_{suffix}/{k}"] = int(v)
                elif isinstance(v, list):
                    synth_dict = synthesize(v)
                    new_logs[f"eval_{suffix}/{k}"] = synth_dict["mean"]
                else:
                    new_logs[f"eval_{suffix}/{k}"] = v
            return new_logs

        wandb.log(modify_logs(argmax_eval_log, "argmax"))
        wandb.log(modify_logs(non_argmax_eval_log, "soft"))

    return_log = {
        "argmax_eval_log": argmax_eval_log,
        "non_argmax_eval_log": non_argmax_eval_log,
    }

    if save_model:
        model_dir = PPO_MODEL_DIR / full_experiment_name
        if not model_dir.exists() or not model_dir.is_dir():
            model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pth"
        torch.save(agent.state_dict(), model_path)
        return model_path, agent, return_log

    return None, agent, return_log


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiment(cfg: DictConfig):
    training_cfg = cfg["training"]
    seed = training_cfg["seed"]

    full_experiment_name = f"{training_cfg['experiment_name']}_{seed}"

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Expect the experiment name to be in the format of
    # taxi_ppo_{model type}_...
    name_list = training_cfg["experiment_name"].split("_")
    # Insert "raw"/"dec" after model type to indicate whether to use
    # the raw observation or the decoded observation
    name_list.insert(3, "dec" if training_cfg["use_decode_obs"] else "raw")
    name_list.append(str(seed))
    full_experiment_name = "_".join(name_list)

    # For wandb and output dir name, capitalise the first 2 words:
    # 'taxi' and 'ppo'
    run_dir_name = "-".join(
        [(s.upper() if i in [0, 1] else s) for i, s in enumerate(name_list)]
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
            monitor_gym=True,
            save_code=True,
            sync_tensorboard=True,
            group=cfg["wandb"]["group"] if "group" in cfg["wandb"] else None,
        )

    # torch.autograd.set_detect_anomaly(True)  # type: ignore

    writer_dir = Path(HydraConfig.get().run.dir) / "tb"
    writer = SummaryWriter(writer_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % (
            "\n".join(
                [
                    f"|{key}|{value}|"
                    for key, value in vars(training_cfg).items()
                ]
            )
        ),
    )

    use_discord_webhook = cfg["webhook"]["use_discord_webhook"]
    msg_body = None
    errored = False
    keyboard_interrupt = None

    try:
        model_path, _, _ = train_ppo(
            training_cfg, full_experiment_name, use_wandb, writer
        )
        assert model_path is not None
        if use_wandb:
            wandb.save(glob_str=str(model_path.absolute()))

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
    import multiprocessing as mp

    if mp.get_start_method() != "forkserver":
        mp.set_start_method("forkserver", force=True)

    run_experiment()
