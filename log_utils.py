import wandb
from typing import Dict

def configure_log(vit_config: Dict, experiment_name: str):
    print(experiment_name)
    if vit_config['log']:
        wandb.config = {
            "learning_rate": vit_config['lr'],
            "experiment_name": experiment_name
        }
        run = wandb.init(project="vit-sigmoid-1", entity="yuvalasher", config=wandb.config)
        return run
    return None
