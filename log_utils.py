import wandb
from typing import Dict

def configure_log(vit_config: Dict, experiment_name: str) -> None:
    if vit_config['log']:
        wandb.config = {
            "learning_rate": vit_config['lr'],
            "experiment_name": experiment_name
        }
        wandb.init(project="my-test-project", entity="yuvalasher", config=wandb.config)
