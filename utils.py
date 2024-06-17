from datetime import datetime
import os
import requests


def post_to_discord_webhook(
    webhook_url: str,
    experiment_name: str,
    message_body: str,
    errored: bool,
    keyboard_interrupt: bool | None = None,
) -> None:
    dt = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    nodename = os.uname().nodename
    if keyboard_interrupt:
        message_head = (
            f"[{dt}]\n"
            f"Experiment {experiment_name} on hine {nodename} "
            f"INTERRUPTED!!\n"
        )
    else:
        message_head = (
            f"[{dt}]\n"
            f"Experiment {experiment_name} on hine {nodename} "
            f"{'ERRORED' if errored else 'FINISHED'}!!\n"
        )

    requests.post(webhook_url, json={"content": message_head + message_body})
