from enum import IntEnum
import json
import logging
from pathlib import Path
import re
import sys
import traceback
from typing import Any

import clingo
import hydra
from omegaconf import DictConfig


file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from utils import post_to_discord_webhook


log = logging.getLogger()


BASE_STORAGE_DIR = root / "dc_ppo_storage"
BASE_ASP_FILE = parent / "base.lp"


class IndividualReturnCode(IntEnum):
    SUCCESS_WITH_STABLE_MODELS = 1
    FAIL_WITH_NO_STABLE_MODELS = -1


def parse_observation(observation: str) -> str:
    # Observation string expected to be in the form of X,Y,O,S
    # where all values are integers
    X, Y, O, S = list(map(int, observation.split(",")))
    location_map: dict[tuple[int, int], str] = {
        (0, 0): "top_left_corner",
        (0, 1): "two_step_ahead",
        (0, 2): "top_right_corner",
        (1, 0): "one_step_ahead_and_left",
        (1, 1): "one_step_ahead",
        (1, 2): "one_step_ahead_and_right",
        (2, 0): "left",
        (2, 1): "curr_location",
        (2, 2): "right",
    }
    object_map: dict[int, str] = {
        0: "unseen",
        1: "empty",
        2: "wall",
        3: "door",
        5: "goal",
    }
    status_map: dict[int, str] = {0: "open", 1: "closed"}

    observation_str_list = [location_map[(X, Y)]]
    if O == 3:
        observation_str_list.append(status_map[S])
    observation_str_list.append(object_map[O])

    return "_".join(observation_str_list)


def parse_stable_model(stable_model: str) -> dict[str, Any]:
    p = re.compile(r"include\(a\((\d+)\),obs\((.+)\),(pos|neg)\)")
    stable_model_list = stable_model.split(" ")

    # Construct the raw mapping
    raw_mapping = {}
    parsed_mapping = {}

    for atom in stable_model_list:
        parsed = p.findall(atom)
        assert len(parsed) == 1, f"Invalid atom: {atom}"
        img_encoding_id_str, obs_str, sign_str = parsed[0]
        img_encoding_id = int(img_encoding_id_str)
        parsed_obs = parse_observation(obs_str)

        if img_encoding_id not in raw_mapping:
            raw_mapping[img_encoding_id] = []
        raw_mapping[img_encoding_id].append((obs_str, sign_str))

        if img_encoding_id not in parsed_mapping:
            parsed_mapping[img_encoding_id] = []
        parsed_body = parsed_obs
        if sign_str == "neg":
            parsed_body = f"not {parsed_obs}"
        parsed_mapping[img_encoding_id].append(parsed_body)

    parsed_program = []
    for i, body_list in parsed_mapping.items():
        parsed_program.append(f"a_{i} :- {', '.join(body_list)}.")

    return {
        "raw_mapping": raw_mapping,
        "parsed_mapping": parsed_mapping,
        "parsed_program": parsed_program,
    }


def interpret(
    model_name: str, base_asp: str, context_asp: str
) -> IndividualReturnCode:
    model_dir = BASE_STORAGE_DIR / model_name
    result_json = model_dir / "interpret_result.json"

    log.info(f"=========={model_name}==========")

    ctl = clingo.Control(["--warn=none", "--opt-mode=optN"])
    ctl.add("base", [], base_asp + context_asp)
    ctl.ground([("base", [])])

    with ctl.solve(yield_=True) as handle:  # type: ignore
        all_answer_sets = [str(a) for a in handle if a.optimality_proven]

    if len(all_answer_sets) == 0:
        log.error("No stable models found.")
        with result_json.open("w") as f:
            json.dump({"num_stable_models": 0}, f)
        return IndividualReturnCode.FAIL_WITH_NO_STABLE_MODELS

    log.info(f"Optimal stable modes count: {len(all_answer_sets)}")

    # Save
    json_dict: dict[Any, Any] = {
        "num_stable_models": len(all_answer_sets),
        "raw_output": all_answer_sets,
    }
    for i, stable_model in enumerate(all_answer_sets):
        json_dict[i] = parse_stable_model(stable_model)
    with result_json.open("w") as f:
        json.dump(json_dict, f, indent=4)

    return IndividualReturnCode.SUCCESS_WITH_STABLE_MODELS


def interpret_multirun(eval_cfg: DictConfig):
    experiment_name = f"{eval_cfg['experiment_name']}"
    use_ndnf = "ndnf" in experiment_name

    assert use_ndnf

    # Check for base ASP file
    assert BASE_ASP_FILE.exists(), "Base ASP file does not exist"
    base_asp = BASE_ASP_FILE.read_text()

    for s in eval_cfg["multirun_seeds"]:
        # Load agent
        model_name = f"{experiment_name}_{s}"
        model_dir = BASE_STORAGE_DIR / model_name
        # Check context ASP file
        context_file_path = model_dir / "context.lp"
        if not context_file_path.exists():
            log.info(
                f"Model {model_dir.name} has no context file. It might have"
                "failed at post-training process, or hasn't been through"
                "context generation yet. Skipping..."
            )
            continue
        context_asp = context_file_path.read_text()

        interpret(model_name, base_asp, context_asp)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run_generate_context(cfg: DictConfig) -> None:
    eval_cfg = cfg["eval"]

    use_discord_webhook = cfg["webhook"]["use_discord_webhook"]
    msg_body = None
    errored = False
    keyboard_interrupt = None

    try:
        return_code = interpret_multirun(eval_cfg)
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
                experiment_name=f"{eval_cfg['experiment_name']} Interpretation",
                message_body=msg_body,
                errored=errored,
                keyboard_interrupt=keyboard_interrupt,
            )


if __name__ == "__main__":
    run_generate_context()
