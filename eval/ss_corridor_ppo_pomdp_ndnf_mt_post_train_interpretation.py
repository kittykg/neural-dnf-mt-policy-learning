# This script interprets the NDNF-MT agent trained on POMDP SpecialStateCorridor
# envs, i.e. using wall status as observation of agent
import json
import logging
from pathlib import Path
import random
import sys
import traceback
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig
import torch


file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass


from ss_corridor_ppo import (
    construct_model,
    construct_single_environment,
    SSCPPONDNFMutexTanhAgent,
)

from eval.ndnf_mt_problog_interpretation import (
    logical_condensation,
    rule_simplification_with_all_possible_states,
    problog_rule_generation,
)
from eval.ss_corridor_ppo_pomdp_ndnf_multirun_eval import (
    SECOND_PRUNE_MODEL_PTH_NAME,
)
from utils import post_to_discord_webhook


DEFAULT_GEN_SEED = 2
DEVICE = torch.device("cpu")
BASE_STORAGE_DIR = root / "ssc_ppo_storage"

log = logging.getLogger()


def interpret(
    model: SSCPPONDNFMutexTanhAgent,
    model_dir: Path,
) -> dict[str, Any]:
    # Load from second prune after thresholding
    # Check for checkpoints first, this should exist
    assert (
        model_dir / SECOND_PRUNE_MODEL_PTH_NAME
    ).exists(), (
        "No 2nd pruned model found, please run the multirun_eval script first"
    )

    pruned_state = torch.load(
        model_dir / SECOND_PRUNE_MODEL_PTH_NAME,
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(pruned_state)
    model.eval()

    # We only operate on the actor
    ndnf_mt = model.actor

    # ========= Interpretation =========
    # Step 1: Raw enumeration of the layers
    # - Compute the bias of the disjunction layer
    abs_disj_weight = torch.abs(ndnf_mt.disjunctions.weights.data)
    # abs_weight: Q x P
    max_abs_disj_w = torch.max(abs_disj_weight, dim=1)[0]
    # max_abs_w: Q
    sum_abs_disj_w = torch.sum(abs_disj_weight, dim=1)
    # sum_abs_w: Q
    disj_bias = sum_abs_disj_w - max_abs_disj_w
    # bias: Q

    # Step 2: Condensation via logical equivalence
    # - Find all the conjunctions that are logically equivalent, i.e. check if
    # the conjunctions are the same.
    condensation_dict = logical_condensation(ndnf_mt)

    # Step 3: Rule simplification based on experienced observations
    # - Based on the observation, compute the truth table based on the used
    # conjunctions, and if any conjunction is always true/false we can remove it
    # from the truth table and replace it with a constant in the weighted logic
    # equation.
    # - This step is optional if we cannot enumerate all possible observations.
    # If we can we can reduce the number of conjunctions required in the
    # weighted logic equations.
    input_tensor = torch.Tensor(
        [
            [-1, -1],  # no wall on either side
            [1, -1],  # wall on the left
            [-1, 1],  # wall on the right
        ],
        device=DEVICE,
    ).float()
    rule_simplification_dict = rule_simplification_with_all_possible_states(
        ndnf_mt, input_tensor, condensation_dict, disj_bias
    )

    # Step 4: Generate ProbLog rules with annotated disjunction based on
    # experienced observations
    # - Compute the probabilities from mutex-tanh output, based on the truth
    # table.
    # - The probabilities are used in the annotated disjunctions head.
    # - The rule body is the entry of the truth table.
    rule_gen_dict = problog_rule_generation(
        rule_simplification_dict, condensation_dict
    )
    problog_rules = rule_gen_dict["problog_rules"]

    return {
        "problog_rules": problog_rules,
        "weighted_logic_equations": rule_simplification_dict[
            "weighted_logic_equations"
        ],
    }


def post_train_interpret(eval_cfg: DictConfig):
    experiment_name = eval_cfg["experiment_name"]
    use_ndnf = "ndnf" in experiment_name

    assert use_ndnf

    use_state_no_as_obs = "sn" in experiment_name
    assert not use_state_no_as_obs, "Has to use wall status as observation"

    single_env = construct_single_environment(eval_cfg)
    num_inputs = single_env.corridor_length if use_state_no_as_obs else 2

    for s in eval_cfg["multirun_seeds"]:
        # Load agent
        model_dir = BASE_STORAGE_DIR / f"{experiment_name}_{s}"
        model = construct_model(
            num_inputs=num_inputs,
            num_latent=eval_cfg["model_latent_size"],
            action_size=int(single_env.action_space.n),
            use_ndnf=use_ndnf,
            use_eo="eo" in experiment_name,
            use_mt="mt" in experiment_name,
        )
        assert isinstance(model, SSCPPONDNFMutexTanhAgent)
        model.to(DEVICE)

        log.info(f"Interpretation of {model_dir.name}:")
        ret = interpret(model, model_dir)
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

    # torch.autograd.set_detect_anomaly(True)

    use_discord_webhook = cfg["webhook"]["use_discord_webhook"]
    msg_body = None
    errored = False
    keyboard_interrupt = None

    try:
        post_train_interpret(eval_cfg)
        if use_discord_webhook:
            msg_body = f"Success!"
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
                experiment_name=f"{eval_cfg['experiment_name']} Multirun Eval",
                message_body=msg_body,
                errored=errored,
                keyboard_interrupt=keyboard_interrupt,
            )


if __name__ == "__main__":
    torch.set_warn_always(False)
    run_eval()
