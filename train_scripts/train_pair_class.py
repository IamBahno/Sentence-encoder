import torch

from datasets import load_dataset
from transformers import BertTokenizer
from functools import partial
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from models.bert_sentence_embedder import BertSentenceEmbedder
from models.pair_class_nli import SimpleClassifier, PairClassNLI
from trainers.trainer_pair_class import Trainer

from dataloader import nli_pair_class_collate_fn as nli_collate_fn


def train_pair_class(cfg):
    dataset = load_dataset(cfg["datasets"]["all-nli"]["name"], "pair-class")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    loss_fn = CrossEntropyLoss()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_loader = DataLoader(dataset["train"], batch_size=cfg["training"]["batch_size"], shuffle=True, collate_fn=partial(nli_collate_fn,tokenizer=tokenizer),drop_last=True)
    val_loader = DataLoader(dataset["dev"], batch_size=cfg["training"]["batch_size"], shuffle=False, collate_fn=partial(nli_collate_fn,tokenizer=tokenizer),drop_last=True)

    model = PairClassNLI(BertSentenceEmbedder(pooling=cfg["embedder"]["pooling"]), SimpleClassifier(input_dim=3*768, output_dim=3)).to(device) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]))

    trainer = Trainer(model, train_loader, val_loader, loss_fn, optimizer, device, cfg["training"]["batch_size"], run_name=cfg["run_name"])
    trainer.train(cfg["training"]["epochs"])