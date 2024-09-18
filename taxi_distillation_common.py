from pathlib import Path
from typing import Any


import numpy as np
import torch
from torch import Tensor
from torch.distributions.categorical import Categorical

from taxi_common import (
    TaxiEnvPPOMLPAgent,
    construct_model,
    split_all_states_to_reachable_and_non,
    taxi_env_preprocess_obs,
)

TAXI_ENV_POSSIBLE_STATES, _ = split_all_states_to_reachable_and_non()


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
