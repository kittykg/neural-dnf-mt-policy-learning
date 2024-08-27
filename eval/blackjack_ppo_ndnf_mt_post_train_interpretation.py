# This script interprets the NDNF-MT actor trained on the Blackjack environment.
# We extract a weighted logic equation and ProbLog rules based on the trained
# model. The weighted logic equation provides insight of the layer itself, while
# the ProbLog rules can be used for inference as well.
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import random
import sys
import traceback
from typing import Any

import numpy as np

import hydra
from omegaconf import DictConfig
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

from blackjack_common import (
    construct_model,
    get_target_policy,
    BlackjackNDNFMutexTanhAgent,
    blackjack_env_preprocess_obss,
    TargetPolicyType,
)
from utils import post_to_discord_webhook


DEFAULT_GEN_SEED = 2
BLACKJACK_SINGLE_ENV_NUM_EPISODES = 500
DEVICE = torch.device("cpu")
BASE_STORAGE_DIR = root / "blackjack_ppo_storage"

FIRST_PRUNE_MODEL_PTH_NAME = "model_soft_mr_pruned.pth"
THRESHOLD_MODEL_PTH_NAME = "soft_thresholded_model.pth"
THRESHOLD_JSON_NAME = "soft_threshold_val_candidates.json"
SECOND_PRUNE_MODEL_PTH_NAME = "model_soft_2nd_mr_pruned.pth"

log = logging.getLogger()


@dataclass
class Atom:
    id: int
    positive: bool
    type: str  # possible values: "input", "conjunction", "disjunction_head"


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
        for i, conjuncts in conjunction_map.items():
            for j, other_conjuncts in conjunction_map.items():
                if i == j:
                    continue
                if (i, j) in duplicate_pairs or (j, i) in duplicate_pairs:
                    continue
                if conjuncts == other_conjuncts:
                    duplicate_pairs.append((i, j))
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
                a.id = duplicate_mapping[a.id]

    used_conjunctions = list(conjunction_map.keys())
    # used conjunctions contain all the conjunctions but remove the duplications
    for i in duplicate_mapping.keys():
        used_conjunctions.remove(i)

    return {
        "conjunction_map": conjunction_map,
        "disjunction_map": disjunction_map,
        "used_conjunctions": used_conjunctions,
        "duplicate_pairs": duplicate_pairs,
        "duplicate_mapping_reverse": duplicate_mapping_reverse,
    }


def rule_simplification(
    ndnf_mt: NeuralDNFMutexTanh,
    target_policy: TargetPolicyType,
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
    preprocessed_obs = blackjack_env_preprocess_obss(
        np.array(list(target_policy.keys())).T, True, DEVICE  # type: ignore
    )
    x = preprocessed_obs["decode_input"]
    x = x.to(DEVICE)

    used_conjunctions = condensation_dict["used_conjunctions"].copy()

    with torch.no_grad():
        truth_table = torch.tanh(ndnf_mt.conjunctions(x)).sign()[
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

            body.append(f"{weight:.4f} conj_{a.id}")

        weighted_logic_equations.append(
            f"{head} = {' + '.join(body)} + {b:.4f}"
        )

    for eq in weighted_logic_equations:
        log.info(eq)

    # Refill the truth_table
    full_conjunction_table = torch.zeros(
        (len(truth_table), ndnf_mt.conjunctions.weights.shape[0]), device=DEVICE
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


def interpret(
    model: BlackjackNDNFMutexTanhAgent,
    target_policy_csv_path: Path,
    model_dir: Path,
) -> dict[str, Any]:
    target_policy = get_target_policy(target_policy_csv_path)

    # Load second prune after thresholding
    # Check for checkpoints first
    assert (
        model_dir / SECOND_PRUNE_MODEL_PTH_NAME
    ).exists(), (
        "Please run the soft extraction first before interpret the model."
    )

    pruned_state = torch.load(
        model_dir / SECOND_PRUNE_MODEL_PTH_NAME, map_location=DEVICE
    )
    model.load_state_dict(pruned_state)
    model.eval()

    # We only operate on the actor
    ndnf_mt = model.actor

    # ========= Interpretation =========
    # Step 1: Raw enumeration of the layers
    # compute the bias of the disjunction layer
    abs_disj_weight = torch.abs(ndnf_mt.disjunctions.weights.data)
    # abs_weight: Q x P
    max_abs_disj_w = torch.max(abs_disj_weight, dim=1)[0]
    # max_abs_w: Q
    sum_abs_disj_w = torch.sum(abs_disj_weight, dim=1)
    # sum_abs_w: Q
    disj_bias = sum_abs_disj_w - max_abs_disj_w
    # bias: Q

    # Step 2: Condensation via logical equivalence
    # Find all the conjunctions that are logically equivalent
    # i.e. check if the conjunctions are the same
    condensation_dict = logical_condensation(ndnf_mt)

    # Step 3: Rule simplification based on experienced observations
    # Based on the observation, compute the truth table based on the used
    # conjunctions, and if any conjunction is always true/false we can remove it
    # from the truth table and replace it with a constant in the weighted logic
    # equation.
    # This step is optional if we cannot enumerate all possible observations. If
    # we can we can reduce the number of conjunctions required in the weighted
    # logic equations.
    rule_simplification_dict = rule_simplification(
        ndnf_mt, target_policy, condensation_dict, disj_bias
    )

    # Step 4: Generate ProbLog rules with annotated disjunction based on
    # experienced observations
    # Compute the probabilities from mutex-tanh output, based on the truth table
    # The probabilities are used in the annotated disjunctions head.
    # The rule body is the entry of the truth table.
    problog_rules = problog_rule_generation(
        rule_simplification_dict, condensation_dict
    )

    return {
        "problog_rules": problog_rules,
        "weighted_logic_equations": rule_simplification_dict[
            "weighted_logic_equations"
        ],
    }


def post_train_eval(eval_cfg: DictConfig):
    experiment_name = f"{eval_cfg['experiment_name']}"
    use_ndnf = "ndnf" in experiment_name

    assert use_ndnf and not eval_cfg["use_eo"] and eval_cfg["use_mt"]

    target_policy_csv_path = Path(eval_cfg["target_policy_csv_path"])
    if not target_policy_csv_path.exists():
        raise FileNotFoundError(
            f"The target policy csv file {target_policy_csv_path} does not exist!"
        )

    for s in eval_cfg["multirun_seeds"]:
        # Load agent
        model_dir = BASE_STORAGE_DIR / f"{experiment_name}_{s}"
        model: BlackjackNDNFMutexTanhAgent = construct_model(
            num_latent=eval_cfg["model_latent_size"],
            use_ndnf=use_ndnf,
            use_decode_obs=True,
            use_eo=False,
            use_mt=True,
            share_layer_with_critic=eval_cfg["share_layer_with_critic"],
        )  # type: ignore
        model.to(DEVICE)

        log.info(f"Interpretation of {model_dir.name}:")
        ret = interpret(model, target_policy_csv_path, model_dir)
        with open(model_dir / "interpretation.json", "w") as f:
            json.dump(ret, f, indent=4)

        with open(model_dir / "problog_rules.pl", "w") as f:
            for r in ret["problog_rules"]:
                f.write(f"{r}\n")

        log.info("======================================")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run_eval(cfg: DictConfig) -> None:
    eval_cfg = cfg["eval"]

    # Set random seed
    torch.manual_seed(DEFAULT_GEN_SEED)
    np.random.seed(DEFAULT_GEN_SEED)
    random.seed(DEFAULT_GEN_SEED)

    torch.autograd.set_detect_anomaly(True)  # type: ignore

    use_discord_webhook = cfg["webhook"]["use_discord_webhook"]
    msg_body = None
    errored = False
    keyboard_interrupt = None

    try:
        post_train_eval(eval_cfg)
        if use_discord_webhook:
            msg_body = f"Success!\n"
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
                experiment_name=eval_cfg["experiment_name"],
                message_body=msg_body,
                errored=errored,
                keyboard_interrupt=keyboard_interrupt,
            )


if __name__ == "__main__":
    torch.set_warn_always(False)
    run_eval()
