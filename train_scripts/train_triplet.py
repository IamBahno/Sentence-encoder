import torch

from datasets import load_dataset
from torch.nn import TripletMarginLoss
from functools import partial
from torch.utils.data import DataLoader
from models.bert_sentence_embedder import BertSentenceEmbedder
from transformers import BertTokenizer

from dataloader import nli_triplet_collate_fn as nli_collate_fn
from trainers.trainer_triplet import Trainer
from train_scripts.optimizer import init_optimizer


def train_triplet(cfg):
    dataset = load_dataset(cfg["datasets"]["all-nli"]["name"], "triplet")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    loss_fn = TripletMarginLoss()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_loader = DataLoader(dataset["train"], batch_size=cfg["training"]["batch_size"], shuffle=True, collate_fn=partial(nli_collate_fn,tokenizer=tokenizer),drop_last=True)
    val_loader = DataLoader(dataset["dev"], batch_size=cfg["training"]["batch_size"], shuffle=False, collate_fn=partial(nli_collate_fn,tokenizer=tokenizer),drop_last=True)

    model = BertSentenceEmbedder(pooling=cfg["embedder"]["pooling"]).to(device)
    optimizer = init_optimizer(model,cfg)
    
    trainer = Trainer(model, train_loader, val_loader, loss_fn, optimizer, device, cfg["training"]["batch_size"], run_name=cfg["run_name"])
    trainer.train(cfg["training"]["epochs"])