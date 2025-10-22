import torch
from torch import nn
from transformers import BertModel

class BertSentenceEmbedder(nn.Module):
    def __init__(self, model_name="bert-base-uncased",pooling="mean"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        assert pooling in ["mean", "max"], f"Unsupported pooling: {pooling}"
        self.pooling = pooling

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

        return sentence_embeddings
    
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
    
class PairClassNLI(nn.Module):
    def __init__(self, embedder, classifier):
        super().__init__()
        self.embedder = embedder
        self.classifier = classifier

    def forward(self, premise_tokens, premise_attention_masks, hypothesis_tokens, hypothesis_attention_masks):
        premise_emb = self.embedder(premise_tokens, premise_attention_masks)
        hypothesis_emb = self.embedder(hypothesis_tokens, hypothesis_attention_masks)
        combined_vector = torch.cat([premise_emb, hypothesis_emb, torch.abs(premise_emb - hypothesis_emb)], dim=-1)

        logits = self.classifier(combined_vector)
        return logits