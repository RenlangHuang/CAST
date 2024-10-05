import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_gather
from einops import rearrange

from models.transformer.output_layer import AttentionOutput
from models.transformer.positional_encoding import RotaryPositionalEmbedding
from models.kpconv import UnaryBlock


class Upsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsampling, self).__init__()
        self.unary = nn.Sequential(
            UnaryBlock(in_channels, out_channels),
            UnaryBlock(out_channels, out_channels)
        )
        self.output = UnaryBlock(out_channels, out_channels)
    
    def forward(self, query, support, upsample_indices):
        """
        Args:
            query (Tensor): (B, N, C)
            support (Tensor): (B, M, C')
            upsample_indices (Tensor): (B, N, 1)
        return:
            latent (Tensor): (B, N, C)
        """
        latent = knn_gather(support, upsample_indices).squeeze(2)
        return self.output(self.unary(latent) + query)


class Downsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsampling, self).__init__()
        self.unary = nn.Sequential(
            UnaryBlock(in_channels, out_channels),
            UnaryBlock(out_channels, out_channels)
        )
        self.output = UnaryBlock(out_channels, out_channels)
    
    def forward(self, q_feats, s_feats, q_points:torch.Tensor, s_points:torch.Tensor, downsample_indices):
        """
        Args:
            q_feats (Tensor): (B, N, C)
            s_feats (Tensor): (B, M, C')
            q_points (Tensor): (B, N, 3)
            s_points (Tensor): (B, N, K, 3)
            downsample_indices (Tensor): (B, N, K)
        return:
            latent (Tensor): (B, M, C)
        """
        grouped_feats = knn_gather(s_feats, downsample_indices) # (B, N, K, C')
        knn_weights = 1. / ((s_points - q_points.unsqueeze(2)).pow(2).sum(-1) + 1e-8) # (B, N, K)
        knn_weights = knn_weights / knn_weights.sum(dim=-1, keepdim=True) # (B, N, K)
        latent = torch.sum(grouped_feats * knn_weights.unsqueeze(-1), dim=2) # (B, N, C)
        return self.output(self.unary(latent) + q_feats)


class SparseTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, pe=True, dropout=None, activation_fn='relu'):
        super(SparseTransformerLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads
        self.pe = pe
        
        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)

        self.linear = nn.Linear(d_model, d_model)
        if dropout is None or dropout <= 0:
            self.dropout = nn.Identity()
        else: self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.output = AttentionOutput(d_model, dropout, activation_fn)
        if pe: self.rpe = RotaryPositionalEmbedding(self.d_model)
    
    @torch.no_grad()
    def select_spots(self, input_knn, memory_knn, confidence_scores, matching_indices, num_spots):
        """
        Args:
            input_knn (Tensor): (B, N, k+1)
            memory_knn (Tensor): (B, M, K)
            confidence_scores (Tensor): (B, N, 1)
            matching_indices (Tensor): (B, N, 1)

        Returns:
            output_states: torch.Tensor (B, N, C)
        """
        knn_scores = knn_gather(confidence_scores, input_knn[...,1:]).squeeze(-1)  # (B, N, k)
        confidence_scores, confident_knn = knn_scores.topk(k=num_spots)  # (B, N, S)
        confident_knn = torch.gather(input_knn[...,1:], -1, confident_knn)  # (B, N, S)
        confident_knn = torch.cat([input_knn[...,:1], confident_knn], dim=-1)  # (B, N, S+1)
        
        spot_indices = knn_gather(matching_indices, confident_knn).squeeze(-1)  # (B, N, S+1)
        spot_indices = knn_gather(memory_knn, spot_indices)  # (B, N, S+1, K)
        spot_indices = rearrange(spot_indices, 'b n s k -> b n (s k)')  # (B, N, (S+1)*K)
        
        # avoid redundant indices from spot areas
        B, N, M = input_knn.shape[0], input_knn.shape[1], memory_knn.shape[1]
        attention_mask = torch.zeros((B, N, M), device=input_knn.device)
        attention_mask.scatter_(-1, spot_indices, 1.)  # (B, N, M)
        spot_mask, spot_indices = attention_mask.topk(spot_indices.shape[-1])  # (B, N, (S+1)*K)
        return spot_mask, spot_indices
    
    def forward(self, input_states, memory_states, indices, input_coord=None, memory_coord=None, attention_mask=None):
        """Sparse Transformer Layer

        Args:
            input_states (Tensor): (B, N, C)
            memory_states (Tensor): (B, M, C)
            indices (Tensor): (B, N, K)
            input_coord (Tensor): (B, N, 3)
            memory_coord (Tensor): (B, M, 3)
            attention_mask (Tensor): (B, N, K)

        Returns:
            output_states: torch.Tensor (B, N, C)
        """
        q = self.proj_q(input_states)  # (B, N, H*C)
        k = knn_gather(self.proj_k(memory_states), indices)  # (B, N, K, H*C)
        v = knn_gather(self.proj_v(memory_states), indices)  # (B, N, K, H*C)
        if self.pe and memory_coord is not None and input_coord is not None:
            k = self.rpe(knn_gather(memory_coord, indices) - input_coord.unsqueeze(2), k)

        q = rearrange(q, 'b n (h c) -> b h n c', h=self.num_heads)
        k = rearrange(k, 'b n m (h c) -> b h n m c', h=self.num_heads)
        v = rearrange(v, 'b n m (h c) -> b h n m c', h=self.num_heads)

        attention_scores = torch.einsum('bhnc,bhnmc->bhnm', q, k) / self.d_model_per_head ** 0.5
        if attention_mask is not None:
            attention_scores = attention_scores - 1e6 * (1. - attention_mask.unsqueeze(1))
        attention_scores = F.softmax(attention_scores, dim=-1)
        hidden_states = torch.sum(attention_scores.unsqueeze(-1) * v, dim=-2)
        hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')
        
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        output_states = self.output(output_states)
        return output_states
