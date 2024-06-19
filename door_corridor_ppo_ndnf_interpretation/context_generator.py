import logging
from pathlib import Path
import re
import sys
import traceback

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


def get_raw_context(
    model: DCPPONDNFBasedAgent, asp_rules: str
) -> dict[str, list] | None:
    obs, _ = single_env.reset()

    terminated = False
    truncated = False
    reward_sum = 0

    all_raw_obs = []
    all_img_encoding = []

    while not terminated and not truncated:
        raw_obs = obs["image"]
        all_raw_obs.append(raw_obs)

        with torch.no_grad():
            raw_img_encoding = model.get_img_encoding(
                preprocessed_obs={
                    "image": torch.tensor(raw_obs.copy(), device=DEVICE)
                    .unsqueeze(0)
                    .float()
                }
            ).squeeze(0)
        img_encoding = torch.nonzero(raw_img_encoding > 0)
        img_encoding_asp = [f"a_{a.item()}." for a in img_encoding]
        all_img_encoding.append(img_encoding)

        ctl = clingo.Control(["--warn=none"])
        show_statements = [
            f"#show disj_{i}/0."
            for i in range(DoorCorridorEnv.get_num_actions())
        ]
        # print(img_encoding_asp + show_statements + asp_rules.split("\n"))
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

        obs, reward, terminated, truncated, _ = single_env.step(action)
        reward_sum += reward

    return {"all_raw_obs": all_raw_obs, "all_img_encoding": all_img_encoding}


def generate_context(
    model: DCPPONDNFBasedAgent, asp_rules: str
) -> list[str] | None:
    raw_context_dict = get_raw_context(model, asp_rules)
    if raw_context_dict is None:
        return None

    asp_context = []
    relevant_encoding_ids = get_relevant_image_encoding_from_rules(asp_rules)

    for i in relevant_encoding_ids:
        asp_context.append(f"img_encoding_id({i}).")

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

    for i in relevant_encoding_ids:
        asp_context.append(
            f"{{ include(a({i}), ({';'.join(obs_lp)}), (pos;neg)) }} 1."
        )

    for time, raw_obs in enumerate(all_raw_obs):
        img_encoding = all_img_encoding[time]
        for a in img_encoding:
            if a.item() in relevant_encoding_ids:
                asp_context.append(
                    f"fired_img_encoding({time}, a({a.item()}))."
                )

        for x in range(3):
            for y in range(3):
                asp_context.append(
                    f"fired_observation({time}, obs({x}, {y}, {raw_obs[x, y, 0]}, {raw_obs[x, y, 1]}))."
                )

    return asp_context


def interpret_image_encoding(eval_cfg: DictConfig):
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
            thresholded_model_path, map_location=DEVICE
        )
        model.load_state_dict(theresholded_model_sd)
        model.eval()

        asp_rules_path = model_dir / "asp_rules.lp"
        assert (
            asp_rules_path.exists()
        ), f"ASP rules not found for {model_dir.name}!"
        with open(model_dir / "asp_rules.lp", "r") as f:
            asp_rules = f.read()

        context = generate_context(model, asp_rules)
        if context is None:
            log.info(f"Failed to generate context for {model_dir.name}!")
            continue

        # Save the context
        log.info(f"Saving context for {model_dir.name}...")
        context_path = model_dir / "context.lp"
        with open(context_path, "w") as f:
            f.write("\n".join(context))


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run_interpret(cfg: DictConfig) -> None:
    eval_cfg = cfg["eval"]

    use_discord_webhook = cfg["webhook"]["use_discord_webhook"]
    msg_body = None
    errored = False
    keyboard_interrupt = None

    try:
        interpret_image_encoding(eval_cfg)
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
    run_interpret()
