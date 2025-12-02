import yaml
import torch
import argparse

from train_scripts.train_mnr import train_mnr
from train_scripts.train_triplet import train_triplet
from train_scripts.train_pair_class import train_pair_class

def load_config(config_path):
    """Load YAML config from given path."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Training script with configurable YAML")
    parser.add_argument("--config", default="config.yaml", 
                       help="Path to config YAML file (default: config.yaml)")
    args = parser.parse_args()
    
    cfg = load_config(args.config)  # Load from CLI arg or default
    

    train_objective = cfg["train_objective"]

    if train_objective == "pair-class":
        train_pair_class(cfg)
    elif train_objective == "triplet":
        train_triplet(cfg)
    elif train_objective == "mnr":
        train_mnr(cfg)
    else:
        raise ValueError(f"Wrong train objective: {train_objective}")

if __name__ == "__main__":
    main()
