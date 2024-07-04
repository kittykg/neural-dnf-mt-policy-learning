from enum import Enum
import gymnasium as gym
import numpy as np
import torch
from torch import Tensor

N_OBSERVATION_SIZE: int = 500
TUPLE_LIMITS_LIST: list[int] = [5, 5, 5, 4]
N_DECODE_OBSERVATION_SIZE: int = sum(TUPLE_LIMITS_LIST)
N_ACTIONS: int = 6

# =============================================================================#
#                               Data structures                                #
# =============================================================================#


class PassengerLocation(Enum):
    RED = (0, (0, 0))
    GREEN = (1, (0, 4))
    YELLOW = (2, (4, 0))
    BLUE = (3, (4, 3))
    TAXI = 4

    @classmethod
    def get_location_from_colour_index(cls, idx: int) -> tuple[int, int]:
        assert idx != cls.TAXI.value, "Taxi is not a colour"

        colour_loc_map: dict[int, tuple[int, int]] = dict(
            [
                cls.RED.value,
                cls.GREEN.value,
                cls.YELLOW.value,
                cls.BLUE.value,
            ]
        )

        return colour_loc_map[idx]


# =============================================================================#
#                             Observation processing                           #
# =============================================================================#


def convert_n_to_one_hot(obs: gym.spaces.Discrete | int) -> np.ndarray:
    """
    Convert the observation into a one-hot encoding.
    """
    return np.eye(N_OBSERVATION_SIZE)[obs]  # type: ignore


def encode(
    taxi_row: int, taxi_col: int, passenger_location: int, destination: int
) -> int:
    """
    Encode the (taxi_row, taxi_col, passenger_location, destination) into a
    single integer.
    """
    for i, j in zip(
        [taxi_row, taxi_col, passenger_location, destination], TUPLE_LIMITS_LIST
    ):
        assert 0 <= i < j, f"{i} going outside of [0, {j})"

    return (
        (taxi_row * 5 + taxi_col) * 5 + passenger_location
    ) * 4 + destination


def decode(obs: gym.spaces.Discrete | int) -> np.ndarray:
    """
    Decode the single integer into a concatenated one-hot encoding of
    (taxi_row, taxi_col, passenger_location, destination).
    """
    i = int(obs)  # type: ignore
    out: list[int] = []
    out.append(i % 4)
    i = i // 4
    out.append(i % 5)
    i = i // 5
    out.append(i % 5)
    i = i // 5
    out.append(i)
    assert 0 <= i < 5

    return np.concatenate(
        [np.eye(s)[o] for o, s in zip(reversed(out), [5, 5, 5, 4])]
    )


# =============================================================================#
#                              Policy comparison                               #
# =============================================================================#


def split_all_states_to_reachable_and_non() -> tuple[list[int], list[int]]:
    """
    In the taxi environment, only 404 out of 500 states are reachable, and 4 of
    the reachable states are terminal states.
    This function split all states into reachable (excluding 4 terminal states,
    400 in total) and non-reachable states (100 in total).
    For more details, see the taxi environment documentation:
    https://gymnasium.farama.org/environments/toy_text/taxi/
    """
    reachable = []
    non_reachable = []

    for x in range(TUPLE_LIMITS_LIST[0]):
        for y in range(TUPLE_LIMITS_LIST[1]):
            for p in range(TUPLE_LIMITS_LIST[2]):
                for d in range(TUPLE_LIMITS_LIST[3]):
                    if p == d:
                        non_reachable.append(encode(x, y, p, d))
                    else:
                        reachable.append(encode(x, y, p, d))

    assert len(reachable) == 400
    assert len(reachable) + len(non_reachable) == 500

    return reachable, non_reachable
