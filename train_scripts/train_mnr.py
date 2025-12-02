import torch

from torch.nn import CrossEntropyLoss
from transformers import BertTokenizer
from functools import partial
from preprocessor import SnliMnliPreprocessor
from torch.utils.data import DataLoader
from models.bert_sentence_embedder import BertSentenceEmbedder
from trainers.trainer_mnr import Trainer
from dataloader import mnr_collate_fn as nli_collate_fn

def train_mnr(cfg):
    dataset_train, dataset_val = SnliMnliPreprocessor(cfg).preprocess()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    loss_fn = CrossEntropyLoss()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_loader = DataLoader(dataset_train, batch_size=cfg["training"]["batch_size"], shuffle=True, collate_fn=partial(nli_collate_fn,tokenizer=tokenizer),drop_last=True)
    val_loader = DataLoader(dataset_val, batch_size=cfg["training"]["batch_size"], shuffle=False, collate_fn=partial(nli_collate_fn,tokenizer=tokenizer),drop_last=True)
   
    model = BertSentenceEmbedder(pooling=cfg["embedder"]["pooling"],cfg=cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]))
    
    trainer = Trainer(model, train_loader, val_loader, loss_fn, optimizer, device, cfg["training"]["batch_size"], run_name=cfg["run_name"])
    trainer.train(cfg["training"]["epochs"])