from collections import OrderedDict

import numpy as np
import numpy.typing as npt
import torch


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(
            m.weight.data.pow(2).sum(1, keepdim=True)
        )
        if m.bias is not None:
            m.bias.data.fill_(0)


def synthesize(array) -> OrderedDict[str, float]:
    d = OrderedDict()
    d["mean"] = np.mean(array)
    d["std"] = np.std(array)
    d["min"] = np.amin(array)
    d["max"] = np.amax(array)
    return d


class DiversityScoreTracker:
    """
    Track the argmax actions count for an agent, and compute the diversity score
    for it.
    """

    possible_actions_count: int
    actions_frequency_dict: dict[int, int]
    total_instances_count: int

    def __init__(self, possible_actions_count: int):
        self.possible_actions_count = possible_actions_count
        self.actions_frequency_dict = {}
        self.total_instances_count = 0

    def update(self, action: npt.NDArray) -> None:
        self.total_instances_count += len(action)

        for a in action:
            if a not in self.actions_frequency_dict:
                self.actions_frequency_dict[a] = 0
            self.actions_frequency_dict[a] += 1

    def compute_action_proportion(self) -> dict[int, float]:
        """
        Compute the proportion of each action in the total actions count.
        p_a = \\frac{\sum_{i \in S} 1[policy(i) == a]}{|S|}
        where i[.] is the indicator function and S is the set of unique
        observations.
        """
        return {
            k: v / self.total_instances_count
            for k, v in self.actions_frequency_dict.items()
        }

    def compute_diversity_score(self) -> float:
        """
        Compute the diversity score for the agent.

        diversity_score(policy)
        = 1 - ( ds1(policy) / (2 - 2/|A|) )
        where ds1(policy) = \\sum_a abs(p_a - 1/|A|) and p_a is the action
        proportion (calculate by `compute_action_proportion`)
        """
        # What we want for the diversity score:
        # 1. Larger means more diverse
        # 2. Smaller means less diverse
        # 3. Fixed minimum of 0: when completely uniform (one action everywhere)
        # 4. Fixed maximum of 1: for when all actions used with equal
        # proportions over states.
        # Optional 5. Cares less about almost perfectly diverse policies
        # difference from perfectly diverse policy, but even slight diversity is
        # distinct from perfectly uniform
        # Optional 6. Simple and easy to explain

        # ds1(policy) = \sum_a abs(p_a - 1/|A|)
        # where p_a is the action proportion (calculate by
        # `compute_action_proportion`)
        # For a perfectly diverse policy, ds1 value is 0.
        # ds1(perfectly diverse policy) = 0
        # For a perfectly uniform policy with no diversity at all, ds1 value is
        # 2 - 2/|A|.
        # ds1(perfectly uniform policy)
        # = (1 - 1/|A|)+ (|A| - 1) * abs(0 - 1/|A|)
        # = 1 - 1/|A| + (|A| - 1) * (1/|A|)
        # = 2 - 2/|A|
        p_a = self.compute_action_proportion()
        card_a = self.possible_actions_count
        equal_appearance_p = 1 / card_a
        ds1 = sum(abs(p - equal_appearance_p) for p in p_a.values())

        # ds2(policy) = 2 - 2/|A| - ds1(policy)
        # ds1(perfectly diverse policy) = 2 - 2/|A|
        # ds1(perfectly uniform policy) = 0
        ds2 = 2 - (2 / card_a) - ds1

        # ds3(policy) = ds2(policy) / (2 - 2/|A|)
        # ds3(perfectly diverse policy) = 1
        # ds3(perfectly uniform policy) = 0
        # This is the final diversity score.
        ds3 = ds2 / (2 - 2 / card_a)

        return ds3

    def compute_entropy(self) -> float:
        action_proportion = self.compute_action_proportion()
        return -sum(p * np.log(p) for p in action_proportion.values() if p != 0)

    def reset(self) -> None:
        self.actions_frequency_dict = {}
        self.total_instances_count = 0
