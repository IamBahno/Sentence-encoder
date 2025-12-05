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
    """
    Expands the pooler to use a weighted average of specific BERT layers.
    It first projects individual layer outputs into one vector (per sample in batch) and then 
    applies the attention pooling on this vector
    """
    def __init__(self, hidden_size=768, last_n_layers=4):
        super().__init__()
        self.layer_indices = [-(i + 1) for i in range(last_n_layers)] # makes list of [-1,-2,...,-n]
        self.layer_indices = self.layer_indices
        self.num_layers = len(self.layer_indices)
        
        # Learnable Weights for layer mixing - one weight per layer
        self.layer_weights = nn.Parameter(torch.ones(self.num_layers))
        # Single layer pooler:
        self.pooler = LastLayerAttentionPooling(hidden_size=hidden_size)

    def forward(self, all_hidden_states, attention_mask):
        """
        all_hidden_states: Tuple of tensors from BERT
                           Each tensor is [B, L, H]
        attention_mask:    [B, L]
        """

        # Stack last n layers
        # Select layers (e.g., last 4)
        # Result shape: [B, num_selected_layers, L, H]
        selected_layers = [all_hidden_states[i] for i in self.layer_indices]
        stacked_layers = torch.stack(selected_layers, dim=1)
        
        # Calculate layer weights
        norm_weights = F.softmax(self.layer_weights, dim=0) # normalization of weights so they sum up to 1
        
        # This allows us to multiply across Batch, L, and H dimensions automatically
        norm_weights = norm_weights.view(1, self.num_layers, 1, 1)
        
        # Weighted sum - squash the "layer" dimension:
        # [B, Layers, L, H] * [1, Layers, 1, 1] -> Sum over Layers -> [B, L, H]
        combined_embedding = (stacked_layers * norm_weights).sum(dim=1)
        
        # Apply the single layer pooler 
        final_embedding = self.pooler(combined_embedding, attention_mask)
        
        return final_embedding