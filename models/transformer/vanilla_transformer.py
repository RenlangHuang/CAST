import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.transformer.output_layer import AttentionOutput


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)

        if dropout is None or dropout <= 0:
            self.dropout = nn.Identity()
        else: self.dropout = nn.Dropout(dropout)

    def forward(self, input_q, input_k, input_v,
                key_weights=None, key_masks=None, attention_factors=None, attention_masks=None):
        """Vanilla attention forward propagation.

        Args:
            input_q (Tensor): input tensor for query (B, N, C)
            input_k (Tensor): input tensor for key (B, M, C)
            input_v (Tensor): input tensor for value (B, M, C)
            key_weights (Tensor): soft masks for the keys (B, M)
            key_masks (BoolTensor): True if ignored, False if preserved (B, M)
            attention_factors (Tensor): factors for attention matrix (B, N, M)
            attention_masks (BoolTensor): True if ignored, False if preserved (B, N, M)

        Returns:
            hidden_states: torch.Tensor (B, C, N)
            attention_scores: intermediate values
                'attention_scores': torch.Tensor (B, H, N, M), attention scores before dropout
        """
        q = rearrange(self.proj_q(input_q), 'b n (h c) -> b h n c', h=self.num_heads)
        k = rearrange(self.proj_k(input_k), 'b m (h c) -> b h m c', h=self.num_heads)
        v = rearrange(self.proj_v(input_v), 'b m (h c) -> b h m c', h=self.num_heads)

        attention_scores = torch.einsum('bhnc,bhmc->bhnm', q, k) / self.d_model_per_head ** 0.5
        if attention_factors is not None:
            attention_scores = attention_factors.unsqueeze(1) * attention_scores
        if key_weights is not None:
            attention_scores = attention_scores * key_weights.unsqueeze(1).unsqueeze(1)
        if key_masks is not None:
            attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1), float('-inf'))
        if attention_masks is not None:
            attention_scores = attention_scores.masked_fill(attention_masks, float('-inf'))
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        hidden_states = torch.matmul(attention_scores, v)
        hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')
        return hidden_states, attention_scores


class AttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(AttentionLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.linear = nn.Linear(d_model, d_model)
        if dropout is None or dropout <= 0:
            self.dropout = nn.Identity()
        else: self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self,
        input_states,
        memory_states,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
        attention_masks=None,
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            memory_states,
            key_weights=memory_weights,
            key_masks=memory_masks,
            attention_factors=attention_factors,
            attention_masks=attention_masks,
        )
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        return output_states, attention_scores


class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, activation_fn='relu'):
        super(TransformerLayer, self).__init__()
        self.attention = AttentionLayer(d_model, num_heads, dropout)
        self.output = AttentionOutput(d_model, dropout, activation_fn)

    def forward(
        self,
        input_states,
        memory_states,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
        attention_masks=None,
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            memory_weights=memory_weights,
            memory_masks=memory_masks,
            attention_factors=attention_factors,
            attention_masks=attention_masks,
        )
        output_states = self.output(hidden_states)
        return output_states, attention_scores
