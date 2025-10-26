import yaml
import torch

from train_scripts.train_mnr import train_mnr
from train_scripts.train_triplet import train_triplet
from train_scripts.train_pair_class import train_pair_class

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

def main():
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
