import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.transformer.output_layer import AttentionOutput


class LinearMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, normalize=True):
        super(LinearMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.normalize = normalize
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)

    def forward(self, input_q, input_k, input_v):
        q = rearrange(self.proj_q(input_q), 'b n (h c) -> b h n c', h=self.num_heads)
        k = rearrange(self.proj_k(input_k), 'b m (h c) -> b h m c', h=self.num_heads)
        v = rearrange(self.proj_v(input_v), 'b m (h c) -> b h m c', h=self.num_heads)
        
        q, k = F.elu(q,1.) + 1., F.elu(k,1.) + 1.
        hidden_states = torch.matmul(q, torch.einsum('bhmc,bhmd->bhcd', k, v))
        if self.normalize:
            hidden_states = hidden_states / (torch.matmul(q, k.sum(2).unsqueeze(-1)) + 1e-4)
        hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')
        return hidden_states


class LinearAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, normalize=True):
        super(LinearAttentionLayer, self).__init__()
        self.attention = LinearMultiHeadAttention(d_model, num_heads, normalize)
        self.linear = nn.Linear(d_model, d_model)
        if dropout is None or dropout <= 0:
            self.dropout = nn.Identity()
        else: self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_states, memory_states):
        hidden_states = self.attention(input_states, memory_states, memory_states)
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        return output_states


class LinearTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, activation_fn='relu', normalize=True):
        super(LinearTransformerLayer, self).__init__()
        self.attention = LinearAttentionLayer(d_model, num_heads, dropout, normalize)
        self.output = AttentionOutput(d_model, dropout, activation_fn)

    def forward(self, input_states, memory_states):
        hidden_states = self.attention(input_states, memory_states)
        output_states = self.output(hidden_states)
        return output_states
