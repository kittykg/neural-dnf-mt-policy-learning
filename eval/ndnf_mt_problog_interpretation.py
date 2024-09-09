from dataclasses import dataclass
import logging
from pathlib import Path
import sys
from typing import Any

import torch
from torch import Tensor

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from neural_dnf import NeuralDNFMutexTanh


log = logging.getLogger()


# This dataclass is the same as the Atom class 'privately' defined in the
# `neural-ndnf` library post-training and is used for ProbLog interpretation
@dataclass
class Atom:
    id: int
    positive: bool
    type: str  # possible values: "input", "conjunction", "disjunction_head"
    map_to_id: int = -1  # used for remove duplication, -1 means no mapping


# ==============================================================================
#                            Interpretation steps                              #
# ==============================================================================

# Step 1: Raw enumeration of the layers
# - Compute the bias of the disjunction layer
# e.g.
# abs_disj_weight = torch.abs(ndnf_mt.disjunctions.weights.data)
# # abs_weight: Q x P
# max_abs_disj_w = torch.max(abs_disj_weight, dim=1)[0]
# # max_abs_w: Q
# sum_abs_disj_w = torch.sum(abs_disj_weight, dim=1)
# # sum_abs_w: Q
# disj_bias = sum_abs_disj_w - max_abs_disj_w
# # bias: Q


# Step 2: Condensation via logical equivalence
# - Find all the conjunctions that are logically equivalent, i.e. check if the
# conjunctions are the same.
# e.g.
# condensation_dict = logical_condensation(ndnf_mt)

# Step 3: Rule simplification based on experienced observations
# - Based on the observation, compute the truth table based on the used
# conjunctions, and if any conjunction is always true/false we can remove it
# from the truth table and replace it with a constant in the weighted logic
# equation.
# - This step is optional if we cannot enumerate all possible observations. If
# we can we can reduce the number of conjunctions required in the weighted
# logic equations.
# e.g.
# input_tensor = ...
# rule_simplification_dict = rule_simplification_with_all_possible_states(
#     ndnf_mt, input_tensor, condensation_dict, disj_bias
# )

# Step 4: Generate ProbLog rules with annotated disjunction based on experienced
# observations
# - Compute the probabilities from mutex-tanh output, based on the truth table
# - The probabilities are used in the annotated disjunctions head.
# - The rule body is the entry of the truth table.
# e.g.
# problog_rules = problog_rule_generation(
#     rule_simplification_dict, condensation_dict
# )


# ==============================================================================
#                            Code implementations                              #
# ==============================================================================


# Step 2: Condensation via logical equivalence
def logical_condensation(ndnf_mt: NeuralDNFMutexTanh) -> dict[str, Any]:
    """
    Condensation via logical equivalence
    Find all the conjunctions that are logically equivalent, i.e. check if the
    conjunctions are the same
    """
    # 1. Extract the skeleton
    conjunction_map: dict[int, list[Atom]] = dict()
    disjunction_map: dict[int, list[Atom]] = dict()
    relevant_input: set[int] = set()

    #       Get all conjunctions
    conj_w = ndnf_mt.conjunctions.weights.data.clone()
    for i, w in enumerate(conj_w):
        if torch.all(w == 0):
            # No conjunction is applied here
            continue

        conjuncts: list[Atom] = []
        for j, v in enumerate(w):
            if v < 0:
                # Negative weight, negate the atom
                conjuncts.append(Atom(j, False, "input"))
            elif v > 0:
                # Positive weight, normal atom
                conjuncts.append(Atom(j, True, "input"))

        conjunction_map[i] = conjuncts

    #       Get the DNF for each class
    disj_w = ndnf_mt.disjunctions.weights.data.clone()
    for i, w in enumerate(disj_w):
        if torch.all(w == 0):
            continue

        disjuncts: list[Atom] = []
        for j, v in enumerate(w):
            if v < 0 and j in conjunction_map:
                # Negative weight, negate the existing conjunction
                disjuncts.append(Atom(j, False, "conjunction"))
                for a in conjunction_map[j]:
                    relevant_input.add(a.id)
            elif v > 0 and j in conjunction_map:
                # Postivie weight, add normal conjunction
                disjuncts.append(Atom(j, True, "conjunction"))
                for a in conjunction_map[j]:
                    relevant_input.add(a.id)

        disjunction_map[i] = disjuncts

    # Check if there are duplicates in the conjunctions
    def check_duplicates(conjunction_map):
        duplicate_pairs = []
        cycle_list = []
        for i, conjuncts in conjunction_map.items():
            i_list = []
            for j, other_conjuncts in conjunction_map.items():
                if i == j:
                    continue
                if (i, j) in duplicate_pairs or (j, i) in duplicate_pairs:
                    continue
                if (i, j) in cycle_list or (j, i) in cycle_list:
                    continue
                if conjuncts == other_conjuncts:
                    i_list.append(j)
            # If i_list is not singleton, we will have 'cycle'
            # For example, id 0 has i_list [1, 3] (dup pairs (0, 1), (0, 3))
            # Then (1, 3) will also be valid duplicated pair, but we don't need
            # to add it to the final duplicate pairs list. So we add it to cycle
            # list to keep duplicate pairs list clean
            if len(i_list) > 1:
                # Compute all possible pairs using i_list
                new_addition_to_cycle_list = []
                for x in i_list:
                    for y in i_list:
                        if x == y:
                            continue
                        if (x, y) in new_addition_to_cycle_list or (
                            y,
                            x,
                        ) in new_addition_to_cycle_list:
                            continue
                        new_addition_to_cycle_list.append((x, y))
                cycle_list.extend(new_addition_to_cycle_list)

            duplicate_pairs.extend([(i, j) for j in i_list])

        return duplicate_pairs

    duplicate_pairs = check_duplicates(conjunction_map)
    # we replace the 2nd conj in the tuple with the 1st element
    duplicate_mapping = {j: i for i, j in duplicate_pairs}

    # reverse
    unique_key = list(set([i for i, _ in duplicate_pairs]))
    duplicate_mapping_reverse = {k: [] for k in unique_key}
    for i, j in duplicate_pairs:
        duplicate_mapping_reverse[i].append(j)

    for _, disjuncts in disjunction_map.items():
        for a in disjuncts:
            if a.id in duplicate_mapping:
                a.map_to_id = duplicate_mapping[a.id]

    used_conjunctions = list(conjunction_map.keys())
    # used conjunctions contain all the conjunctions but remove the duplications
    for i in duplicate_mapping.keys():
        used_conjunctions.remove(i)

    return {
        "conjunction_map": conjunction_map,
        "disjunction_map": disjunction_map,
        "used_conjunctions": used_conjunctions,
        "duplicate_pairs": duplicate_pairs,
        "duplicate_mapping": duplicate_mapping,
        "duplicate_mapping_reverse": duplicate_mapping_reverse,
    }


# Step 3: Rule simplification based on experienced observations
def rule_simplification_with_all_possible_states(
    ndnf_mt: NeuralDNFMutexTanh,
    input_tensor: Tensor,
    condensation_dict: dict[str, Any],
    disjunction_bias: Tensor,
) -> dict[str, Any]:
    """
    In this step, we compute the weight logic program, and the probability
    for generating ProbLog annotated disjunctions
    """
    # Compute the truth table
    # Since we know all the possible states, we can compute the truth table
    # with them
    used_conjunctions = condensation_dict["used_conjunctions"].copy()
    duplicate_mapping = condensation_dict["duplicate_mapping"]

    with torch.no_grad():
        truth_table = torch.tanh(ndnf_mt.conjunctions(input_tensor)).sign()[
            :, used_conjunctions
        ]

    # Remove any duplicate entries in the truth table
    truth_table = truth_table.unique(dim=0)

    # Check if any conjunction is always true or always false
    always_true = []
    always_false = []

    for i, t in enumerate(truth_table.T):
        conj_id = list(condensation_dict["conjunction_map"].keys())[i]
        if torch.all(t == 1):
            log.info(f"Conjunction {conj_id} is always true")
            always_true.append(conj_id)
        if torch.all(t == -1):
            log.info(f"Conjunction {conj_id} is always false")
            always_false.append(conj_id)

    # Generate the weighted logic equations
    weighted_logic_equations = []

    for disj_id, disjuncts in condensation_dict["disjunction_map"].items():
        head = f"action({disj_id})"
        body = []
        b = disjunction_bias[disj_id]

        for a in disjuncts:
            weight = ndnf_mt.disjunctions.weights[disj_id, a.id]
            if a.id in always_true:
                b += weight
                continue

            if a.id in always_false:
                b += -1 * weight
                continue

            a_id = a.id
            if a.id in duplicate_mapping and a.map_to_id != -1:
                assert a.map_to_id == duplicate_mapping[a.id]
                a_id = duplicate_mapping[a.id]

            body.append(f"{weight:.4f} conj_{a_id}")

        weighted_logic_equations.append(
            f"{head} = {' + '.join(body)} + {b:.4f}"
        )

    for eq in weighted_logic_equations:
        log.info(eq)

    # Refill the truth_table
    full_conjunction_table = torch.zeros(
        (len(truth_table), ndnf_mt.conjunctions.weights.shape[0]),
        device=input_tensor.device,
    )
    duplicate_mapping_reverse = condensation_dict["duplicate_mapping_reverse"]

    for i, t in enumerate(truth_table):
        for j, v in enumerate(t):
            conj_id = used_conjunctions[j]
            full_conjunction_table[i, conj_id] = v

            if conj_id in duplicate_mapping_reverse:
                for mapped_to_id in duplicate_mapping_reverse[conj_id]:
                    full_conjunction_table[i, mapped_to_id] = v

    with torch.no_grad():
        mt_out = ndnf_mt.disjunctions(full_conjunction_table)
    prob = (mt_out + 1) / 2

    return {
        "truth_table": truth_table,
        "used_conjunctions": used_conjunctions,
        "always_true": always_true,
        "always_false": always_false,
        "weighted_logic_equations": weighted_logic_equations,
        "prob": prob,
    }


# Step 4: Generate ProbLog rules with annotated disjunction based on experienced
def problog_rule_generation(
    rule_simplification_dict: dict[str, Any], condensation_dict: dict[str, Any]
) -> list[str]:
    """
    Generate ProbLog rules with annotated disjunction based on experienced
    observations.
    Return a list of ProbLog rules.
    """

    def cast_probabilities_to_3_decimal(prob) -> list[float]:
        new_prob = []
        for i in range(len(prob) - 1):
            p = prob[i]
            new_prob.append(round(p.item(), 3))
        last_p = 1 - sum(new_prob)
        new_prob.append(last_p)
        return new_prob

    truth_table = rule_simplification_dict["truth_table"]
    prob = rule_simplification_dict["prob"]
    used_conjunctions = rule_simplification_dict["used_conjunctions"]
    always_true = rule_simplification_dict["always_true"]
    always_false = rule_simplification_dict["always_false"]

    # Compute pure problog rules
    problog_rules = []
    for i, entry in enumerate(truth_table):
        rule_head = []
        three_decimal_prob = cast_probabilities_to_3_decimal(prob[i])
        for disj_id in range(prob.shape[1]):
            rule_head.append(
                f"{three_decimal_prob[disj_id]:.3f}::action({disj_id})"
            )
        rule_head = " ; ".join(rule_head)

        rule_body = []
        for j, v in enumerate(entry):
            conj_id = used_conjunctions[j]
            if conj_id in always_true or conj_id in always_false:
                continue
            if v == 1:
                rule_body.append(f"conj_{conj_id}")
            elif v == -1:
                rule_body.append(f"\+conj_{conj_id}")  # type: ignore

        rule_body = ", ".join(rule_body)
        problog_rules.append(f"{rule_head} :- {rule_body}.")

    conjunction_map: dict[int, list[Atom]] = condensation_dict[
        "conjunction_map"
    ]
    for conj_id, conjuncts in conjunction_map.items():
        if conj_id in always_true or conj_id in always_false:
            continue

        if conj_id not in used_conjunctions:
            continue

        rule_head = f"conj_{conj_id}"
        rule_body = []
        for a in conjuncts:
            if a.positive:
                rule_body.append(f"input({a.id})")
            else:
                rule_body.append(f"\+input({a.id})")  # type: ignore
        rule_body = ", ".join(rule_body)
        problog_rules.append(f"{rule_head} :- {rule_body}.")

    for r in problog_rules:
        log.info(r)

    return problog_rules
