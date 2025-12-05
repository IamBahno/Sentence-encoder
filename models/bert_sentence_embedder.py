import torch
from torch import nn
from transformers import BertModel
from models.attention_pooling import LastLayerAttentionPooling,MultiLayerAttentionPooling

class BertSentenceEmbedder(nn.Module):
    def __init__(self, model_name="bert-base-uncased",pooling="mean",cfg=None):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        assert pooling in ["mean", "max", "attention_last_layer", "attention_multi_layer"], f"Unsupported pooling: {pooling}"
        self.pooling = pooling
        if pooling == "attention_last_layer":
            self.pooling_layer = LastLayerAttentionPooling(num_heads=cfg["heads"],num_queries_per_head=cfg["queries_per_head"])
        elif pooling == "attention_multi_layer":
            self.bert = BertModel.from_pretrained(model_name,output_hidden_states=True)
            self.single_layer_pooling = LastLayerAttentionPooling(num_heads=cfg["heads"],num_queries_per_head=cfg["queries_per_head"])
            self.pooling_layer = MultiLayerAttentionPooling(last_n_layers=cfg["multi_layer_pooling"]["last_n_layers"])


    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # (batch, seq_len, hidden)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())

        if self.pooling == "mean":
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            token_counts = input_mask_expanded.sum(1)
            # avoid division by zero, if empty sentence
            token_counts = torch.clamp(token_counts, min=1e-9)    
            sentence_embeddings = sum_embeddings / token_counts    
        elif self.pooling == "max":
            # Max pooling (mask padded positions with very negative value so they don't dominate)
            token_embeddings[input_mask_expanded == 0] = -1e9
            sentence_embeddings = torch.max(token_embeddings, dim=1).values
        elif self.pooling == "attention_last_layer":
            sentence_embeddings = self.pooling_layer(token_embeddings,attention_mask)
        elif self.pooling == "attention_multi_layer":
            sentence_embeddings = self.pooling_layer(outputs.hidden_states,attention_mask) # if it is multi layer the bert also returns hidden_states
        else:
            print("Incorrect pooling")
            exit()
        return sentence_embeddings