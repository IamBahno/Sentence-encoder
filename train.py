import os
import torch

from datasets import load_dataset
from transformers import BertTokenizer
from functools import partial


from utils.plot import plot_losses
from torch.utils.data import DataLoader
from torch.nn import TripletMarginLoss
from model import BertSentenceEmbedder
from trainer import Trainer
from dataloader import nli_triplet_collate_fn





def train():

    run_name = "triplet_max_pooling_bert"

    dataset = load_dataset("sentence-transformers/all-nli",'triplet')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)


    loss_fn = TripletMarginLoss(margin=1.0, p=2)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    batch_size=16
    epochs = 20


    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, collate_fn=partial(nli_triplet_collate_fn,tokenizer=tokenizer),drop_last=True)
    val_loader = DataLoader(dataset["dev"], batch_size=batch_size, shuffle=False, collate_fn=partial(nli_triplet_collate_fn,tokenizer=tokenizer),drop_last=True)


    model = BertSentenceEmbedder(pooling="max").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)


    trainer = Trainer(model,train_loader,val_loader,loss_fn,optimizer,device,batch_size,run_name=run_name)
    trainer.train(epochs)


if __name__ == "__main__":
    
    # TODO since we test on benchmark, we can add testing data to val or train
    # TODO also check setings 'pair-class', 'pair-score','triplet', 'pair'
    # TODO add some config file, but atleast do so command line args 
    # TODO maybe logger instead of printing
    train()