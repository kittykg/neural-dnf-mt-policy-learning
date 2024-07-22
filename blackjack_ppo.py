import logging
from pathlib import Path
import random
import time
import traceback
from typing import Any


import gymnasium as gym
import hydra
from hydra.core.hydra_config import HydraConfig
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter  # type: ignore
import wandb


from neural_dnf.utils import DeltaDelayedExponentialDecayScheduler

from blackjack_common import *
from common import DiversityScoreTracker
from utils import post_to_discord_webhook


PPO_MODEL_DIR = Path(__file__).parent / "blackjack_ppo_storage/"
if not PPO_MODEL_DIR.exists() or not PPO_MODEL_DIR.is_dir():
    PPO_MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TARGET_STATE = (16, 5, 1)
log = logging.getLogger()


def get_relevant_targets_from_target_policy(
    use_ndnf: bool,
    target_policy: TargetPolicyType,
    device: torch.device,
    normalise: bool = False,
) -> dict[str, Any]:
    """
    Return the observations, preprocessed observations and target actions from a
    target policy trained from tabular Q-learning.
    """
    obs_list = [obs for obs in target_policy.keys()]
    target_q_actions = np.array([target_policy[obs] for obs in obs_list])

    input_np_arr = np.stack(
        [non_decode_obs(obs, normalise) for obs in obs_list]
    )
    decode_input_nd_array = np.stack(
        [decode_tuple_obs(obs) for obs in obs_list]
    )

    if use_ndnf:
        decode_input_nd_array = np.where(
            decode_input_nd_array == 0, -1, decode_input_nd_array
        )

    obs_dict = {
        "input": torch.tensor(
            input_np_arr,
            dtype=torch.float32,
            device=device,
        ),
        "decode_input": torch.tensor(
            decode_input_nd_array,
            dtype=torch.float32,
            device=device,
        ),
    }

    return {
        "obs_list": obs_list,
        "obs_dict": obs_dict,
        "target_q_actions": target_q_actions,
    }


def get_agent_policy(
    agent: BlackjackBaseAgent,
    target_q_policy: TargetPolicyType,
    device: torch.device,
    normalise: bool = False,
) -> Any:
    """
    Return the action distribution for the agent at all states presented in
    `target_q_policy`.
    """
    obs_dict = get_relevant_targets_from_target_policy(
        isinstance(agent, BlackjackNDNFBasedAgent),
        target_q_policy,
        device,
        normalise,
    )["obs_dict"]

    with torch.no_grad():
        action_dist = agent.get_action_distribution(obs_dict)

    return action_dist.probs.cpu().numpy()  # type: ignore


def track_target_state_action_distribution(
    agent: BlackjackBaseAgent,
    device: torch.device,
    target_state: tuple[int, int, int] = DEFAULT_TARGET_STATE,
) -> float:
    """
    Return the probability of HIT at `target_state` for the agent.
    """
    input_np_arr = np.array([list(target_state)])
    decode_input_nd_array = np.stack([decode_tuple_obs(target_state)])

    if isinstance(agent, BlackjackNDNFBasedAgent):
        decode_input_nd_array = np.where(
            decode_input_nd_array == 0, -1, decode_input_nd_array
        )

    obs_dict = {
        "input": torch.tensor(
            input_np_arr,
            dtype=torch.float32,
            device=device,
        ),
        "decode_input": torch.tensor(
            decode_input_nd_array,
            dtype=torch.float32,
            device=device,
        ),
    }
    with torch.no_grad():
        action_dist = agent.get_action_distribution(obs_dict)

    return action_dist.probs.cpu().numpy()[0, 1]  # type: ignore


def plot_policy_grid_after_train(
    target_policy_csv_path: Path,
    agent: BlackjackBaseAgent,
    device: torch.device,
    model_name: str,
    use_wandb: bool,
) -> None:
    """
    Plot the agent policy grid after training
    """
    target_policy = get_target_policy(target_policy_csv_path)
    action_distribution = get_agent_policy(agent, target_policy, device)
    plot = create_policy_plots_from_action_distribution(
        target_policy,
        action_distribution,
        model_name,
        argmax=True,
        plot_diff=True,
    )
    plot.savefig(f"{model_name}_argmax_policy_cmp_q.png")
    if use_wandb:
        wandb.log(
            {
                "argmax_policy": wandb.Image(
                    f"{model_name}_argmax_policy_cmp_q.png"
                )
            }
        )
    plt.close()

    plot = create_policy_plots_from_action_distribution(
        target_policy,
        action_distribution,
        model_name,
        argmax=False,
    )
    plot.savefig(f"{model_name}_soft_policy_cmp_q.png")
    if use_wandb:
        wandb.log(
            {"soft_policy": wandb.Image(f"{model_name}_soft_policy_cmp_q.png")}
        )
    plt.close()


def mlp_agent_cmp_target_csv(
    target_policy_csv_path: Path,
    agent: BlackjackMLPAgent,
    device: torch.device,
    normalise: bool = False,
) -> dict[str, Any]:
    logs: dict[str, Any] = {}

    target_policy = get_target_policy(target_policy_csv_path)
    ret = get_relevant_targets_from_target_policy(
        False, target_policy, device, normalise
    )
    obs_dict = ret["obs_dict"]
    target_q_actions = ret["target_q_actions"]

    dst = DiversityScoreTracker(N_ACTIONS)
    with torch.no_grad():
        actions = agent.get_actions(obs_dict)
        dst.update(actions)

    policy_error_cmp_to_q = np.count_nonzero(actions != target_q_actions) / len(
        target_q_actions
    )
    logs["policy_error_cmp_to_q"] = policy_error_cmp_to_q
    logs["action_diversity_score"] = dst.compute_diversity_score()
    logs["action_entropy"] = dst.compute_entropy()

    return logs


def ndnf_based_agent_cmp_target_csv(
    target_policy_csv_path: Path,
    agent: BlackjackNDNFBasedAgent,
    device: torch.device,
) -> dict[str, Any]:
    logs: dict[str, Any] = {
        "mutual_exclusivity": True,
        "missing_actions": False,
    }

    target_policy = get_target_policy(target_policy_csv_path)
    ret = get_relevant_targets_from_target_policy(True, target_policy, device)
    obs_list = ret["obs_list"]
    obs_dict = ret["obs_dict"]
    target_q_actions = ret["target_q_actions"]

    dst = DiversityScoreTracker(N_ACTIONS)
    with torch.no_grad():
        actions, tanh_actions = agent.get_actions(
            preprocessed_obs=obs_dict, use_argmax=True
        )
        dst.update(actions)

    tanh_actions_discretised = np.count_nonzero(tanh_actions > 0, axis=1)
    if np.any(tanh_actions_discretised > 1):
        logs["mutual_exclusivity"] = False
        logs["mutual_exclusivity_violations_count"] = int(
            np.count_nonzero(tanh_actions_discretised > 1)
        )
        logs["mutual_exclusivity_violations_states"] = [
            obs_list[i] for i in np.where(tanh_actions_discretised > 1)[0]
        ]

    if np.any(tanh_actions_discretised == 0):
        logs["missing_actions"] = True
        logs["missing_actions_count"] = int(
            np.count_nonzero(tanh_actions_discretised == 0)
        )
        logs["missing_actions_states"] = [
            obs_list[i] for i in np.where(tanh_actions_discretised == 0)[0]
        ]

    policy_error_cmp_to_q = np.count_nonzero(actions != target_q_actions) / len(
        target_q_actions
    )
    logs["policy_error_cmp_to_q"] = policy_error_cmp_to_q
    logs["action_diversity_score"] = dst.compute_diversity_score()
    logs["action_entropy"] = dst.compute_entropy()
    log.info(dst.compute_action_proportion())

    return logs


def train_ppo(
    training_cfg: DictConfig,
    full_experiment_name: str,
    use_wandb: bool,
    writer: SummaryWriter,
    save_model: bool = True,
) -> tuple[Path | None, BlackjackBaseAgent, dict[str, Any] | None]:
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
        [make_env(i, i, False) for i in range(training_cfg["num_envs"])]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    # Set up the model
    share_layer_with_critic = training_cfg.get("share_layer_with_critic", False)

    agent = construct_model(
        num_latent=training_cfg["model_latent_size"],
        use_ndnf=use_ndnf,
        use_decode_obs=use_decode_obs,
        use_eo="use_eo" in training_cfg and training_cfg["use_eo"],
        use_mt="use_mt" in training_cfg and training_cfg["use_mt"],
        share_layer_with_critic=share_layer_with_critic,
    )
    agent.train()
    agent.to(device)

    optimizer = optim.Adam(
        agent.parameters(), lr=training_cfg["learning_rate"], eps=1e-5
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
    delta_one_count = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_obs_dict = blackjack_env_preprocess_obss(next_obs, use_ndnf, device)
    next_done = torch.zeros(num_envs).to(device)

    last_episodic_return = None

    if isinstance(agent, BlackjackNDNFBasedAgent):
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
            lr_now = frac * training_cfg["learning_rate"]
            optimizer.param_groups[0]["lr"] = lr_now

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
            rewards[step] = torch.tensor(reward).float().to(device).view(-1)
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
        b_obs = obs.reshape((-1, num_inputs))
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

        if isinstance(agent, BlackjackNDNFBasedAgent):
            delta_dict = dds.step(agent.actor)
            new_delta = delta_dict["new_delta_vals"][0]
            old_delta = delta_dict["old_delta_vals"][0]

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    training_cfg = cfg["training"]
    seed = training_cfg["seed"]

    full_experiment_name = f"{training_cfg['experiment_name']}_{seed}"

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # For wandb and output dir name, capitalise the first 2 words:
    # 'blackjack' and 'ppo'
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
