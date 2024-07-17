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

    actions_count_dict: dict[int, int]
    total_actions_count: int

    def __init__(self):
        self.actions_count_dict = {}
        self.total_actions_count = 0

    def update(self, action: npt.NDArray) -> None:
        self.total_actions_count += len(action)

        for a in action:
            if a not in self.actions_count_dict:
                self.actions_count_dict[a] = 0
            self.actions_count_dict[a] += 1

    def compute_diversity_score(self) -> float:
        """
        Compute the diversity score for the agent.

        The diversity score is defined as the sum of the difference between the
        action proportion and the uniform distribution.
        """
        action_proportion = self.compute_action_proportion()
        uniform_distribution = 1 / len(action_proportion)
        return sum(
            abs(p - uniform_distribution) for p in action_proportion.values()
        )

    def compute_entropy(self) -> float:
        action_proportion = self.compute_action_proportion()
        return -sum(p * np.log(p) for p in action_proportion.values() if p != 0)

    def compute_action_proportion(self) -> dict[int, float]:
        return {
            k: v / self.total_actions_count
            for k, v in self.actions_count_dict.items()
        }

    def reset(self) -> None:
        self.actions_count_dict = {}
        self.total_actions_count = 0
