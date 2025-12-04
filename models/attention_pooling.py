import torch
import torch.nn as nn
import torch.nn.functional as F


class LastLayerAttentionPooling(nn.Module):
    """
    Multi-head attention pooling:
        Inputs: token_embeddings [B, L, H], attention_mask [B, L]
        Q: learnable query (per head)
        K,V: projections from token embeddings
        Output: single pooled embedding [B, H]
    """
    def __init__(self, hidden_size=768, num_heads=8, num_queries_per_head=32):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_queries_per_head = num_queries_per_head

        assert hidden_size % num_heads == 0

        # Learned queries: [heads, Q_per_head, d]
        self.query = nn.Parameter(
            torch.randn(num_heads, num_queries_per_head, self.head_dim)
        )

        # K and V projections
        self.key_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # MLP maps flattened pooled representation → final vector
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * num_queries_per_head, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

    def forward(self, token_embeddings, attention_mask):
        """
        token_embeddings: [B, L, H]
        attention_mask:   [B, L]  (1 for valid tokens, 0 for padding)
        """
        B, L, H = token_embeddings.size()

        # Project K, V → [B, heads, L, d]
        K = self.key_proj(token_embeddings)
        V = self.value_proj(token_embeddings)

        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Expand Q: [B, heads, Q_ph, d] (it will like stack the same tensor Q batch times)
        Q = self.query.unsqueeze(0).expand(B, -1, -1, -1)

        # Attention: [B, heads, Q, L]
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)

        # Apply mask
        mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)  # [B,1,1,L]
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)

        # Weighted sum: [B, heads, Q, d]
        pooled = torch.matmul(attn_weights, V)

        # Flatten
        pooled = pooled.reshape(B, H * self.num_queries_per_head)
        # Final output
        return self.mlp(pooled)


class MultiLayerAttentionPooling(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self,hidden_outputs,attention_mask):
        return hidden_outputs