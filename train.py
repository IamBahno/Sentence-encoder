import torch
import yaml

from datasets import load_dataset
from transformers import BertTokenizer
from functools import partial

from torch.utils.data import DataLoader

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

if cfg["dataset"]["subset"] == "pair-class":
    from torch.nn import CrossEntropyLoss as loss_func
    from dataloader import nli_pair_class_collate_fn as nli_collate_fn
    from models import BertSentenceEmbedder, SimpleClassifier, PairClassNLI
    from trainer_pair_class import Trainer

if cfg["dataset"]["subset"] == "triplet":
    from torch.nn import TripletMarginLoss as loss_func
    from dataloader import nli_triplet_collate_fn as nli_collate_fn
    from models import BertSentenceEmbedder
    from trainer_triplet import Trainer


def train():
    dataset = load_dataset(cfg["dataset"]["name"], cfg["dataset"]["subset"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    loss_fn = loss_func()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_loader = DataLoader(dataset["train"], batch_size=cfg["training"]["batch_size"], shuffle=True, collate_fn=partial(nli_collate_fn,tokenizer=tokenizer),drop_last=True)
    val_loader = DataLoader(dataset["dev"], batch_size=cfg["training"]["batch_size"], shuffle=False, collate_fn=partial(nli_collate_fn,tokenizer=tokenizer),drop_last=True)
   
    if cfg["dataset"]["subset"] == "triplet":
        model = BertSentenceEmbedder(pooling=cfg["embedder"]["pooling"]).to(device)
    elif cfg ["dataset"]["subset"] == "pair-class":
        model = PairClassNLI(BertSentenceEmbedder(pooling=cfg["embedder"]["pooling"]), SimpleClassifier(input_dim=3*768, output_dim=3)).to(device) 

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]))
    
    trainer = Trainer(model, train_loader, val_loader, loss_fn, optimizer, device, cfg["training"]["batch_size"], run_name=cfg["run_name"])
    trainer.train(cfg["training"]["epochs"])

if __name__ == "__main__":
    # TODO since we test on benchmark, we can add testing data to val or train
    # TODO also check setings 'pair-class', 'pair-score','triplet', 'pair'
    # TODO add some config file, but atleast do so command line args 
    # TODO maybe logger instead of printing

    train()