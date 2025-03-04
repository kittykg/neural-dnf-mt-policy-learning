import logging
from pathlib import Path
import re
import sys
import traceback
from typing import Any

import clingo
import hydra
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

from corridor_grid.envs import DoorCorridorEnv
from door_corridor_ppo import construct_model, DCPPONDNFBasedAgent
from utils import post_to_discord_webhook

Atom = tuple[int, bool, str]

log = logging.getLogger()

single_env = DoorCorridorEnv(render_mode="rgb_array")
DEVICE = torch.device("cpu")
BASE_STORAGE_DIR = root / "dc_ppo_storage"


def get_relevant_image_encoding_from_rules(asp_rules: str):
    """
    Return the list of indices of the image encoding that are relevant, i.e.
    being used in the rule bodies.
    Expecting the ASP rules to be in a single string
    """
    return list(
        set(int(r.split("_")[-1]) for r in re.findall(r"a_\d+", asp_rules))
    )


def get_disjunction_map(sd: dict[str, Any]) -> dict[str, dict[int, list[Atom]]]:
    conjunction_map: dict[int, list[Atom]] = dict()
    disjunction_map: dict[int, list[Atom]] = dict()

    conj_w = sd["conjunctions.weights"]
    for i, w in enumerate(conj_w):
        if torch.all(w == 0):
            # No conjunction is applied here
            continue

        conjuncts: list[Atom] = []
        for j, v in enumerate(w):
            if v < 0:
                # Negative weight, negate the atom
                conjuncts.append((j, False, "input"))
            elif v > 0:
                # Positive weight, normal atom
                conjuncts.append((j, True, "input"))

        conjunction_map[i] = conjuncts

    disj_w = sd["disjunctions.weights"]
    for i, w in enumerate(disj_w):
        if torch.all(w == 0):
            # No DNF for class i
            # This should not happen
            continue

        disjuncts: list[Atom] = []
        for j, v in enumerate(w):
            if v < 0 and j in conjunction_map:
                # Negative weight, negate the existing conjunction
                disjuncts.append((j, False, "conjunction"))
            elif v > 0 and j in conjunction_map:
                # Postivie weight, add normal conjunction
                disjuncts.append((j, True, "conjunction"))

        disjunction_map[i] = disjuncts

    return {
        "conjunction_map": conjunction_map,
        "disjunction_map": disjunction_map,
    }


def get_raw_context(
    model: DCPPONDNFBasedAgent, asp_rules: str
) -> dict[str, Any] | None:
    obs, _ = single_env.reset()

    ret = get_disjunction_map(model.actor.state_dict())
    conjunction_map = ret["conjunction_map"]
    disjunction_map = ret["disjunction_map"]

    # Construct the action selection rules at each timestep.
    # This is mutual exclusivity check for each action at each timestep.
    action_check_rules = []
    for i in disjunction_map.keys():
        action_check_rules.append(f"disj_id({i}).")
        for j, sign, _ in disjunction_map[i]:
            body_str = "" if sign else "not "
            body_str += f"conj_{j}(T)"
            action_check_rules.append(
                f"disj(T, {i}) :- timestamp(T), {body_str}."
            )

    for i, conjs in conjunction_map.items():
        body_list = []
        for j, sign, _ in conjs:
            body_str = "" if sign else "-"
            body_str += f"a_fired_at_timestep(T, a({j}))"
            body_list.append(body_str)
        action_check_rules.append(
            f"conj_{i}(T) :-\n    " + ",\n    ".join(body_list) + "."
        )

    # Compute at each timestep, what image encodings are fired and what
    # observations are fired
    terminated = False
    truncated = False

    all_raw_obs = []
    all_img_encoding = []

    time = 0
    while not terminated and not truncated:
        raw_obs = obs["image"]
        all_raw_obs.append(raw_obs)

        with torch.no_grad():
            preprocessed_obs = {
                "image": torch.tensor(raw_obs.copy(), device=DEVICE)
                .unsqueeze(0)
                .float()
            }
            raw_img_encoding = model.get_img_encoding(
                preprocessed_obs=preprocessed_obs
            ).squeeze(0)

        img_encoding = torch.nonzero(raw_img_encoding > 0)
        img_encoding_asp = [f"a_{a.item()}." for a in img_encoding]
        all_img_encoding.append(img_encoding)

        ctl = clingo.Control(["--warn=none"])
        show_statements = [
            f"#show disj_{i}/0."
            for i in range(DoorCorridorEnv.get_num_actions())
        ]

        ctl.add(
            "base",
            [],
            " ".join(
                img_encoding_asp + show_statements + asp_rules.split("\n")
            ),
        )
        ctl.ground([("base", [])])
        with ctl.solve(yield_=True) as handle:  # type: ignore
            all_answer_sets = [str(a) for a in handle]

        if len(all_answer_sets) != 1:
            # No model or multiple answer sets, should not happen
            log.info(f"No model or multiple answer sets when evaluating rules.")
            return None

        if all_answer_sets[0] == "":
            log.info(f"No output action!")
            return None

        output_classes = all_answer_sets[0].split(" ")
        if len(output_classes) == 0:
            log.info(f"No output action!")
            return None

        output_classes_set = set([int(o[5:]) for o in output_classes])

        if len(output_classes_set) != 1:
            log.info(f"Output set: {output_classes_set} not exactly one item!")
            return None

        action = list(output_classes_set)[0]

        obs, _, terminated, truncated, _ = single_env.step(action)
        time += 1

    return {
        "action_check_rules": action_check_rules,
        "all_raw_obs": all_raw_obs,
        "all_img_encoding": all_img_encoding,
    }


def generate_context_from_single_model_and_asp(
    model: DCPPONDNFBasedAgent, asp_rules: str
) -> list[str] | None:
    raw_context_dict = get_raw_context(model, asp_rules)
    if raw_context_dict is None:
        return None

    asp_context = []
    relevant_encoding_ids = get_relevant_image_encoding_from_rules(asp_rules)

    for i in relevant_encoding_ids:
        asp_context.append(f"img_encoding_id({i}).")

    action_check_rules = raw_context_dict["action_check_rules"]
    all_raw_obs = raw_context_dict["all_raw_obs"]
    all_img_encoding = raw_context_dict["all_img_encoding"]
    total_time = len(all_img_encoding)

    asp_context.append(f"timestamp(0..{total_time - 1}).")

    # Generate the obs/4 facts
    obs_lp = set()
    for raw_obs in all_raw_obs:
        for x in range(3):
            for y in range(3):
                obs_lp.add(
                    f"obs({x}, {y}, {raw_obs[x, y, 0]}, {raw_obs[x, y, 1]})"
                )

    for obs in obs_lp:
        asp_context.append(f"is_possible_observation({obs}).")

    # Choice rule
    for i in relevant_encoding_ids:
        asp_context.append(
            f"1 {{ include(a({i}), ({';'.join(obs_lp)}), (pos;neg)) }}."
        )

    # The context at each time step
    for time, raw_obs in enumerate(all_raw_obs):
        img_encoding = all_img_encoding[time]
        # What img encodings are fired at this time
        for a in img_encoding:
            if a.item() in relevant_encoding_ids:
                asp_context.append(
                    f"fired_img_encoding_gt({time}, a({a.item()}))."
                )

        # What observations are fired at this time
        for x in range(3):
            for y in range(3):
                asp_context.append(
                    f"fired_obs_at_timestep({time}, obs({x}, {y}, {raw_obs[x, y, 0]}, {raw_obs[x, y, 1]}))."
                )

    # The action check rules
    asp_context.extend(action_check_rules)

    return asp_context


def generate_context_multirun(eval_cfg: DictConfig):
    experiment_name = f"{eval_cfg['experiment_name']}"
    use_ndnf = "ndnf" in experiment_name

    assert use_ndnf

    for s in eval_cfg["multirun_seeds"]:
        # Load agent
        model_dir = BASE_STORAGE_DIR / f"{experiment_name}_{s}"
        model = construct_model(
            eval_cfg,
            DoorCorridorEnv.get_num_actions(),
            use_ndnf,
            single_env.observation_space["image"],  # type: ignore
        )
        assert isinstance(model, DCPPONDNFBasedAgent)
        model.to(DEVICE)

        # Check for checkpoints: required thresholded model and ASP rules
        thresholded_model_path = model_dir / "thresholded_model.pth"
        if not thresholded_model_path.exists():
            log.info(
                f"Model {model_dir.name} either has not gone through "
                "post-training evaluation or failed to have threshold candidate"
                " at post-training evaluation!"
            )
            continue

        theresholded_model_sd = torch.load(
            thresholded_model_path, map_location=DEVICE, weights_only=True
        )
        model.load_state_dict(theresholded_model_sd)
        model.eval()

        asp_rules_path = model_dir / "asp_rules.lp"
        assert (
            asp_rules_path.exists()
        ), f"ASP rules not found for {model_dir.name}!"
        with open(model_dir / "asp_rules.lp", "r") as f:
            asp_rules = f.read()

        context = generate_context_from_single_model_and_asp(model, asp_rules)
        if context is None:
            log.info(f"Failed to generate context for {model_dir.name}!")
            continue

        # Save the context
        log.info(f"Saving context for {model_dir.name}...")
        context_path = model_dir / "context.lp"
        with open(context_path, "w") as f:
            f.write("\n".join(context))


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run_generate_context(cfg: DictConfig) -> None:
    eval_cfg = cfg["eval"]

    use_discord_webhook = cfg["webhook"]["use_discord_webhook"]
    msg_body = None
    errored = False
    keyboard_interrupt = None

    try:
        generate_context_multirun(eval_cfg)
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
                experiment_name=f"{eval_cfg['experiment_name']} Context Generation",
                message_body=msg_body,
                errored=errored,
                keyboard_interrupt=keyboard_interrupt,
            )


if __name__ == "__main__":
    run_generate_context()
