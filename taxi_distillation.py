import logging
from pathlib import Path
import random
import traceback
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from torch import Tensor
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader, Dataset
import wandb

from neural_dnf.neural_dnf import (
    BaseNeuralDNF,
    NeuralDNF,
    NeuralDNFEO,
    BaseNeuralDNFMutexTanh,
    NeuralDNFMutexTanh,
    NeuralDNFFullMutexTanh,
)
from neural_dnf.utils import DeltaDelayedExponentialDecayScheduler
from eval.taxi_distillation_rl_eval_common import (
    eval_on_environments,
    eval_on_all_possible_states,
)
from taxi_common import (
    N_OBSERVATION_SIZE,
    N_DECODE_OBSERVATION_SIZE,
    N_ACTIONS,
    split_all_states_to_reachable_and_non,
    TaxiEnvPPOMLPAgent,
    construct_model,
    taxi_env_preprocess_obs,
)
from utils import post_to_discord_webhook

log = logging.getLogger()


TAXI_ENV_POSSIBLE_STATES, _ = split_all_states_to_reachable_and_non()
STORAGE_DIR = Path(__file__).parent / "taxi_distillation_storage"


class TaxiDistillationDataset(Dataset):
    """
    The dataset stores (state in one-hot encoding, action distribution) pair
    """

    train: bool
    use_decode_obs: bool
    output_as_action_distribution: bool

    distilled_input: Tensor  # state in one-hot encoding
    distilled_output: Tensor  # action distribution or action index

    def __init__(
        self,
        distilled_input: Tensor,
        distilled_output: Tensor,
        train: bool,
        use_decode_obs: bool,
        output_as_action_distribution: bool,
        repeat: int = 1,
    ) -> None:
        super().__init__()
        self.train = train
        self.use_decode_obs = use_decode_obs
        self.output_as_action_distribution = output_as_action_distribution

        self.distilled_input = torch.cat([distilled_input] * repeat, dim=0)
        self.distilled_output = torch.cat([distilled_output] * repeat, dim=0)

    def __len__(self) -> int:
        return len(self.distilled_input)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        i = self.distilled_input[index]
        # Check if all 0 are converted to -1
        i = torch.where(i == 0, -1, i)
        o = self.distilled_output[index]
        return i, o


def load_mlp_model(
    model_architecture_cfg: dict[str, Any],
    mlp_model_path_str: str,
    device: torch.device,
) -> TaxiEnvPPOMLPAgent:
    model = construct_model(**model_architecture_cfg)
    assert isinstance(model, TaxiEnvPPOMLPAgent)

    model.to(device)
    mlp_model_path = Path(mlp_model_path_str)
    assert mlp_model_path.exists(), f"Path {mlp_model_path} does not exist"

    sd = torch.load(mlp_model_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model


def generate_data_from_mlp(
    model: TaxiEnvPPOMLPAgent, device: torch.device
) -> tuple[dict[str, Tensor], Categorical]:
    """
    Generate data from the PPO MLP agent
    Returns:
        preprocessed_obs: preprocessed observations
        act_dist: action distribution from the PPO MLP agent
    """
    preprocessed_obs = taxi_env_preprocess_obs(
        np.array(TAXI_ENV_POSSIBLE_STATES), use_ndnf=False, device=device
    )
    with torch.no_grad():
        act_dist = model.get_action_distribution(preprocessed_obs)

    return preprocessed_obs, act_dist


def load_target_q_table(tab_q_path_str: str) -> np.ndarray:
    tab_q_path = Path(tab_q_path_str)
    assert tab_q_path.exists(), f"Path {tab_q_path} does not exist"
    tab_q = np.load(tab_q_path)

    return tab_q[TAXI_ENV_POSSIBLE_STATES]


def generate_data_from_tab_q(
    tab_q_path_str: str, device: torch.device
) -> tuple[dict[str, Tensor], Tensor]:
    """
    Generate data from the Q table
    Returns:
        preprocessed_obs: preprocessed observations for NDNF-based models
        q_values: Q values from the Q table
    """
    q_values = load_target_q_table(tab_q_path_str)
    preprocessed_obs = taxi_env_preprocess_obs(
        np.array(TAXI_ENV_POSSIBLE_STATES), use_ndnf=True, device=device
    )
    return preprocessed_obs, torch.tensor(q_values).float()


def get_dataloaders(
    distilled_input: Tensor,
    distilled_output: Tensor,
    batch_size: int,
    use_decode_obs: bool,
    output_as_action_distribution: bool,
    repeat: int = 1,
    num_workers: int = 4,
    pin_memory: bool = False,
    train_loader_persistent_workers: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Get the training and testing dataloaders
    """
    train_dataset = TaxiDistillationDataset(
        distilled_input,
        distilled_output,
        train=True,
        use_decode_obs=use_decode_obs,
        output_as_action_distribution=output_as_action_distribution,
        repeat=repeat,
    )
    test_dataset = TaxiDistillationDataset(
        distilled_input,
        distilled_output,
        train=False,
        use_decode_obs=use_decode_obs,
        output_as_action_distribution=output_as_action_distribution,
        repeat=1,
    )
    return (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=train_loader_persistent_workers,
        ),
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    )


# Weight auxiliary loss for sparsity
def _weight_reg_aux_loss(ndnf_model: BaseNeuralDNF):
    p_t = torch.cat(
        [
            parameter.view(-1)
            for parameter in ndnf_model.parameters()
            if parameter.requires_grad
        ]
    )
    return torch.abs(p_t * (6 - torch.abs(p_t))).mean()


def _loss_calculation_mt(
    ndnf_model: BaseNeuralDNFMutexTanh,
    input: Tensor,
    target: Tensor,
    train_cfg: DictConfig,
    criterion: torch.nn.Module,
) -> dict[str, Tensor]:
    # Activation loss for MT
    def mt_activation_loss(y: Tensor):
        # push each of the raw logits to be close to 6/-6 while
        # maintaining its mutex-tanh property

        # | y^2 - 36 |
        # return torch.abs(y**2 - 36).sum()

        # |36 - min(|y|, 6)^2|
        return torch.abs(
            36 - torch.minimum(torch.abs(y), torch.ones(y.shape) * 6) ** 2
        ).sum()

    all_forms_dict = ndnf_model.get_all_forms(input)

    probs = (all_forms_dict["disjunction"]["mutex_tanh"] + 1) / 2
    if isinstance(criterion, torch.nn.KLDivLoss):
        # KL div expects input to be log probabilities
        log_probs = torch.log(probs + 1e-8)
        base_loss = criterion(log_probs, target)
    else:
        base_loss = criterion(probs, target)

    # Weight regularization loss
    weight_reg_loss = _weight_reg_aux_loss(ndnf_model)

    # Auxiliary loss to push mutex tanh and tanh close to each other
    tanh_out = all_forms_dict["disjunction"]["tanh"]
    p_k_hat = (tanh_out + 1) / 2
    l_ce1 = -torch.sum(probs * torch.log(p_k_hat + 1e-8))
    l_ce2 = -torch.sum((1 - probs) * torch.log(1 - p_k_hat + 1e-8))

    loss = (
        base_loss
        + weight_reg_loss * train_cfg["aux_loss"]["weight_reg_lambda"]
        + (l_ce1 + l_ce2) * train_cfg["aux_loss"]["mt_lambda"]
    )

    return_dict = {
        "loss": loss,
        "base_loss": base_loss,
        "weight_reg_loss": weight_reg_loss,
        "l_ce1": l_ce1,
        "l_ce2": l_ce2,
    }

    # ----- The following loss might not be activated -----
    # Auxiliary loss for mutex-tanh activations
    disj_act_aux_loss_lambda = train_cfg["aux_loss"].get(
        "disj_act_aux_loss_lambda", None
    )
    if disj_act_aux_loss_lambda is not None:
        disj_act_aux_loss = mt_activation_loss(
            all_forms_dict["disjunction"]["mutex_tanh"]
        )
        loss += disj_act_aux_loss * disj_act_aux_loss_lambda
        return_dict["disj_act_aux_loss"] = disj_act_aux_loss

    if isinstance(ndnf_model, NeuralDNFFullMutexTanh):
        conj_act_aux_loss_lambda = train_cfg["aux_loss"].get(
            "conj_act_aux_loss_lambda", None
        )
        if conj_act_aux_loss_lambda is not None:
            conj_act_aux_loss = mt_activation_loss(
                all_forms_dict["conjunction"]["mutex_tanh"]
            )
            loss += conj_act_aux_loss * conj_act_aux_loss_lambda
            return_dict["conj_act_aux_loss"] = conj_act_aux_loss

        # Also pushes the conjunctions MT out and tanh out to be
        # close to each other if the model is NeuralDNFFullMutexTanh
        conj_probs = (all_forms_dict["conjunction"]["mutex_tanh"] + 1) / 2
        conj_tanh_out = all_forms_dict["conjunction"]["tanh"]
        conj_p_k_hat = (conj_tanh_out + 1) / 2
        l_conj_ce1 = -torch.sum(conj_probs * torch.log(conj_p_k_hat + 1e-8))
        l_conj_ce2 = -torch.sum(
            (1 - conj_probs) * torch.log(1 - conj_p_k_hat + 1e-8)
        )

        loss += (l_conj_ce1 + l_conj_ce2) * train_cfg["aux_loss"]["mt_lambda"]

        return_dict["l_conj_ce1"] = l_conj_ce1
        return_dict["l_conj_ce2"] = l_conj_ce2

    return return_dict


def _loss_calculation_plain_and_eo(
    ndnf_model: NeuralDNF | NeuralDNFEO,
    input: Tensor,
    target: Tensor,
    train_cfg: DictConfig,
    criterion: torch.nn.Module,
) -> dict[str, Tensor]:
    # Plain NeuralDNF or NeuralDNF-EO
    probs = ndnf_model(input)
    base_loss = criterion(probs, target)
    weight_reg_loss = _weight_reg_aux_loss(ndnf_model)

    # Conjunction activation loss for plain and EO
    conj_out = torch.tanh(ndnf_model.conjunctions(input))
    conj_out_reg = (1 - conj_out.abs()).mean()

    loss = (
        base_loss
        + weight_reg_loss * train_cfg["aux_loss"]["weight_reg_lambda"]
        + conj_out_reg * train_cfg["aux_loss"]["conj_reg_lambda"]
    )

    return {
        "loss": loss,
        "base_loss": base_loss,
        "weight_reg_loss": weight_reg_loss,
        "conj_out_reg_loss": conj_out_reg,
    }


def train(
    train_cfg: DictConfig, use_wandb: bool
) -> tuple[BaseNeuralDNF, dict[str, Any]]:
    use_decode_obs = train_cfg["use_decode_obs"]
    use_argmax_action = train_cfg["use_argmax_action"]
    model_type_str = train_cfg["model_type"]

    if model_type_str == "eo" or model_type_str == "plain":
        assert use_argmax_action, "argmax action must be True for EO and plain"

    use_cuda = torch.cuda.is_available() and train_cfg["use_cuda"]
    use_mps = (
        torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
        and train_cfg.get("use_mps", False)
    )
    if use_mps:
        device = torch.device("mps")
    else:
        device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )

    target_q_table = None
    target_action_dist = None

    if train_cfg["distillation_mlp"]["mlp_model_path"] is not None:
        # Pass a dummy config to load_model
        distillation_mlp_cfg: dict[str, Any] = OmegaConf.to_container(
            train_cfg["distillation_mlp"].copy()
        )  # type: ignore
        mlp_model_path_str = distillation_mlp_cfg.pop("mlp_model_path")
        mlp_model = load_mlp_model(
            model_architecture_cfg=distillation_mlp_cfg,
            mlp_model_path_str=mlp_model_path_str,
            device=device,
        )

        preprocessed_obs, target_action_dist = generate_data_from_mlp(
            mlp_model, device
        )
        distilled_output = (
            target_action_dist.probs.max(dim=1)[1]  # type: ignore
            if use_argmax_action
            else target_action_dist.probs
        )
    else:
        assert (
            train_cfg["distillation_tab_q"]["tab_q_path"] is not None
        ), "Either mlp_model_path or tab_q_path must be provided"

        assert (
            use_argmax_action
        ), "argmax action must be True when using Q table for distillation"

        tab_q_path_str = train_cfg["distillation_tab_q"]["tab_q_path"]
        target_q_table = load_target_q_table(tab_q_path_str)
        preprocessed_obs, q_values = generate_data_from_tab_q(
            tab_q_path_str, device
        )
        distilled_output = torch.argmax(q_values, dim=1)

    obs_key = "decode_input" if use_decode_obs else "input"
    train_loader, _ = get_dataloaders(
        distilled_input=preprocessed_obs[obs_key],
        distilled_output=distilled_output,  # type: ignore
        batch_size=train_cfg["batch_size"],
        use_decode_obs=use_decode_obs,
        output_as_action_distribution=not use_argmax_action,
        repeat=train_cfg.get("repeat", 1),
    )

    model_type: BaseNeuralDNF = {
        "plain": NeuralDNF,
        "eo": NeuralDNFEO,
        "mt": NeuralDNFMutexTanh,
        "fmt": NeuralDNFFullMutexTanh,
    }[model_type_str]
    ndnf_model: BaseNeuralDNF = model_type(
        num_preds=(
            N_DECODE_OBSERVATION_SIZE if use_decode_obs else N_OBSERVATION_SIZE
        ),
        num_conjuncts=train_cfg["num_conjunctions"],
        n_out=N_ACTIONS,
        delta=1.0,
        weight_init_type=train_cfg.get("weight_init_type", "normal"),
    )
    ndnf_model.to(device)
    ndnf_model.train()
    log.info(ndnf_model)

    dds = DeltaDelayedExponentialDecayScheduler(
        initial_delta=train_cfg["dds"]["initial_delta"],
        delta_decay_delay=train_cfg["dds"]["delta_decay_delay"],
        delta_decay_steps=train_cfg["dds"]["delta_decay_steps"],
        delta_decay_rate=train_cfg["dds"]["delta_decay_rate"],
        target_module_type=ndnf_model.__class__.__name__,
    )
    ndnf_model.set_delta_val(train_cfg["dds"]["initial_delta"])

    optimizer = torch.optim.Adam(  # type: ignore
        ndnf_model.parameters(), lr=train_cfg["lr"]
    )

    if isinstance(ndnf_model, NeuralDNFEO):
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion_fn_map = {
            "kl": torch.nn.KLDivLoss(reduction="batchmean"),
            "mse": torch.nn.MSELoss(),
            "ce": torch.nn.CrossEntropyLoss(),
        }
        criterion_fn_str = train_cfg.get("criterion_fn", "kl")
        assert (
            criterion_fn_str in criterion_fn_map
        ), "Invalid criterion function"

        criterion = criterion_fn_map[criterion_fn_str]

    for epoch in range(int(train_cfg["epoch"])):
        loss_history = None
        for _, (input, target) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            input = input.to(device)

            if isinstance(ndnf_model, BaseNeuralDNFMutexTanh):
                loss_dict = _loss_calculation_mt(
                    ndnf_model, input, target, train_cfg, criterion
                )
            else:
                loss_dict = _loss_calculation_plain_and_eo(
                    ndnf_model, input, target, train_cfg, criterion  # type: ignore
                )

            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()

            if loss_history is None:
                loss_history = loss
            else:
                loss_history += loss

        # For now we only support changing the delta in all layers at the same
        # time
        delta_dict = dds.step(ndnf_model)

        new_delta = delta_dict["new_delta_vals"][0]
        old_delta = delta_dict["old_delta_vals"][0]

        if use_wandb:
            assert isinstance(loss_history, Tensor), "loss_history is None"

            base_loss = loss_dict["base_loss"]
            weight_reg_loss = loss_dict["weight_reg_loss"]
            log_dict = {
                "epoch": epoch,
                "loss": np.mean(loss_history.cpu().detach().numpy()).item(),
                "new_delta": new_delta,
                "old_delta": old_delta,
                "base_loss": base_loss.item(),
                "weight_reg_loss": weight_reg_loss,
            }
            if isinstance(ndnf_model, BaseNeuralDNFMutexTanh):
                l_ce1 = loss_dict["l_ce1"]
                l_ce2 = loss_dict["l_ce2"]

                log_dict["l_ce1"] = l_ce1.item()
                log_dict["l_ce2"] = l_ce2.item()

                if "disj_act_aux_loss" in loss_dict:
                    act_loss = loss_dict["disj_act_aux_loss"]
                    log_dict["disj_act_aux_loss"] = act_loss.item()

                if isinstance(ndnf_model, NeuralDNFFullMutexTanh):
                    l_conj_ce1 = loss_dict["l_conj_ce1"]
                    l_conj_ce2 = loss_dict["l_conj_ce2"]

                    log_dict["l_conj_ce1"] = l_conj_ce1.item()
                    log_dict["l_conj_ce2"] = l_conj_ce2.item()

                    if "conj_act_aux_loss_lambda" in loss_dict:
                        mt_conj_act_loss = loss_dict["conj_act_aux_loss_lambda"]
                        log_dict["conj_act_aux_loss_lambda"] = (
                            mt_conj_act_loss.item()
                        )
            else:
                conj_out_reg = loss_dict["conj_out_reg_loss"]
                log_dict["conj_out_reg_loss"] = conj_out_reg.item()

            wandb.log(log_dict)

        if epoch % 100 == 0:
            assert isinstance(loss_history, Tensor), "loss_history is None"

            base_loss = loss_dict["base_loss"]
            weight_reg_loss = loss_dict["weight_reg_loss"]

            log.info(
                f"Epoch [{epoch:3d}]\tloss: "
                f"{np.mean(loss_history.cpu().detach().numpy()).item():.3f}"
                f"\tdelta used: {new_delta:.3f}"
            )
            detailed_loss_info_str = f"base_loss: {base_loss.item():.3f}\t"
            detailed_loss_info_str += (
                f"weight_reg_loss: {weight_reg_loss:.3f}\t"
            )

            if isinstance(ndnf_model, BaseNeuralDNFMutexTanh):
                l_ce1 = loss_dict["l_ce1"]
                l_ce2 = loss_dict["l_ce2"]

                detailed_loss_info_str += (
                    f"l_ce1: {l_ce1.item():.3f}\tl_ce2: {l_ce2.item():.3f}"
                )

                if "act_loss" in loss_dict:
                    act_loss = loss_dict["act_loss"]
                    detailed_loss_info_str += (
                        f"\tact_loss: {act_loss.item():.3f}"
                    )
            else:
                conj_out_reg = loss_dict["conj_out_reg_loss"]
                detailed_loss_info_str += (
                    f"conj_out_reg_loss: {conj_out_reg.item():.3f}"
                )

            log.info(detailed_loss_info_str)

    post_train_eval_log = {}
    log1 = eval_on_environments(ndnf_model, device)
    log2 = eval_on_all_possible_states(
        ndnf_model, device, target_q_table, target_action_dist
    )
    post_train_eval_log.update(log1)
    post_train_eval_log.update(log2)

    log.info(post_train_eval_log)

    if use_wandb:
        wandb_log = {}
        for k, v in post_train_eval_log.items():
            if isinstance(v, bool):
                wandb_log[f"eval/{k}"] = int(v)
            elif isinstance(v, (list, Tensor, np.ndarray)):
                continue
            else:
                wandb_log[f"eval/{k}"] = v
        wandb.log(wandb_log)

    return ndnf_model, post_train_eval_log


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    train_cfg = cfg["training"]
    seed = train_cfg["seed"]
    if seed is None:
        seed = random.randint(0, 10000)
    full_experiment_name = train_cfg["experiment_name"] + f"_{seed}"

    use_wandb = cfg["wandb"]["use_wandb"]

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    run_dir_name = "-".join(
        [
            (s.upper() if i in [0, 1] else s)
            for i, s in enumerate(full_experiment_name.split("_"))
        ]
    )

    if use_wandb:
        run = wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"]["entity"],
            config=OmegaConf.to_container(train_cfg),  # type: ignore
            dir=HydraConfig.get().run.dir,
            name=run_dir_name,
            tags=cfg["wandb"]["tags"] if "tags" in cfg["wandb"] else [],
            group=cfg["wandb"]["group"] if "group" in cfg["wandb"] else None,
        )

    torch.autograd.set_detect_anomaly(True)  # type: ignore

    use_discord_webhook = cfg["webhook"]["use_discord_webhook"]
    msg_body = None
    errored = False
    keyboard_interrupt = None

    try:
        ndnf_model, eval_log = train(train_cfg, use_wandb)
        model_dir = STORAGE_DIR / full_experiment_name
        if not model_dir.exists() or not model_dir.is_dir():
            model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pth"
        torch.save(ndnf_model.state_dict(), model_path)

        if use_discord_webhook:
            msg_body = "Success!\n"
            for k, v in eval_log.items():
                if isinstance(v, (list, Tensor, np.ndarray)):
                    continue
                msg_body += f"\t{k}: {v}\n"
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
