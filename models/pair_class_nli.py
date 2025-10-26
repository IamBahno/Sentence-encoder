import torch
from torch import nn

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