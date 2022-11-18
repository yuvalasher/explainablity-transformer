import wandb
from typing import Dict
from config import config


def configure_log(vit_config: Dict, experiment_name: str, image_name: str):
    print(experiment_name)
    if vit_config["log"]:
        wandb.config = {
            "learning_rate": vit_config["lr"],
            "experiment_name": experiment_name,
            "image_name": image_name,
        }
        run = wandb.init(project="vit-sigmoid-1", entity=config["general"]["wandb_entity"], config=wandb.config)
        return run
    return None


def get_wandb_config(vit_config: Dict, experiment_name: str, image_name: str):
    if vit_config["log"]:
        wandb.config = {
            "learning_rate": vit_config["lr"],
            "experiment_name": experiment_name,
            "image_name": image_name,
        }
        return wandb.config
    return None
