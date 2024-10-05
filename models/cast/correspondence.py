import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points, knn_gather

from models.kpconv import UnaryBlock
from models.transformer.vanilla_transformer import AttentionLayer


class KeypointMatching(nn.Module):
    def __init__(self, d_embed, num_neighbors, learnable=True):
        super(KeypointMatching, self).__init__()
        self.num_neighbors = num_neighbors
        self.learnable = learnable
        if self.learnable:
            self.proj_q = nn.Linear(d_embed, d_embed, bias=False)
            self.proj_k = nn.Linear(d_embed, d_embed, bias=False)

            self.W = torch.nn.Parameter(torch.zeros(d_embed, d_embed), requires_grad=True)
            torch.nn.init.normal_(self.W, std=0.1)
        else:
            self.proj_q, self.proj_k = nn.Identity(), nn.Identity()
            self.W = torch.nn.Parameter(torch.eye(d_embed) * 0.5, requires_grad=False)

    def forward(self, feat, knn_xyz, knn_feat:torch.Tensor, weights=None, knn_weights=None, knn_mask:torch.Tensor=None):
        q = self.proj_q(feat).unsqueeze(1)  # (N, 1, C)
        k = self.proj_k(knn_feat).transpose(1, 2)  # (N, C, K)
        attention_scores = torch.matmul(q, k).squeeze(1)  # (N, K)
        if knn_mask is not None:
            attention_scores = attention_scores - (~knn_mask).float() * 1e12
        
        if self.num_neighbors > 0 and self.num_neighbors < k.shape[-1]:
            neighbor_mask = torch.full_like(attention_scores, fill_value=float('-inf'))
            neighbor_mask[:, torch.topk(attention_scores, k=self.num_neighbors, dim=-1)[1]] = 0
            attention_scores = attention_scores + neighbor_mask
        
        attention_scores = torch.softmax(attention_scores, dim=-1)  # (N, K)
        corres_xyz = torch.einsum('nk,nkc->nc', attention_scores, knn_xyz)  # (N, 3)
        corres_feat = torch.einsum('nk,nkc->nc', attention_scores, knn_feat)  # (N, C)
        
        W_triu = torch.triu(self.W)
        W_symmetrical = W_triu + W_triu.T
        match_logits = torch.einsum('nc,cd,nkd->nk', feat, W_symmetrical, knn_feat)  # (N, K)
        logit = torch.einsum('nc,cd,nd->n', feat, W_symmetrical, corres_feat).unsqueeze(-1)  # (N, 1)
        if knn_weights is None:
            attentive_feats = torch.cat([feat, corres_feat, logit], dim=-1) # (N, 2C+1)
        else:
            corres_weight = torch.sum(attention_scores * knn_weights.squeeze(-1), dim=-1, keepdim=True)  # (N, 1)
            attentive_feats = torch.cat([feat, corres_feat, weights.unsqueeze(-1), corres_weight, logit], dim=-1) # (N, 2C+3)
        return corres_xyz, attentive_feats, match_logits


class FineMatching(nn.Module):
    def __init__(self, d_embed, num_neighbors, max_distance, learnable=True):
        super(FineMatching, self).__init__()
        self.k = num_neighbors 
        self.max_dist = max_distance
        if learnable:
            self.W = torch.nn.Parameter(torch.zeros(d_embed, d_embed), requires_grad=True)
            torch.nn.init.normal_(self.W, std=0.1)
        else:
            self.W = torch.nn.Parameter(torch.eye(d_embed) * 0.5, requires_grad=False)
    
    def forward(self, ref_points:torch.Tensor, src_points:torch.Tensor, ref_feats:torch.Tensor, src_feats:torch.Tensor):
        dist, knn_indices, knn_xyz = knn_points(src_points.unsqueeze(0), ref_points.unsqueeze(0), K=self.k, return_nn=True)
        weight = torch.relu(1. - dist.squeeze() / self.max_dist)  # (N, K) or (N,) (k=1)
        if self.k == 1:  # not learnable
            return knn_xyz.squeeze(), weight  # (N, 3), (N,)
        
        W_triu = torch.triu(self.W)
        W_symmetrical = W_triu + W_triu.T
        knn_feats = knn_gather(ref_feats.unsqueeze(0), knn_indices).squeeze(0)  # (N, K, C)
        attention_scores = torch.einsum('nc,cd,nkd->nk', src_feats, W_symmetrical, knn_feats)  # (N, K)
        attention_scores = torch.softmax(attention_scores * weight, dim=-1)  # (N, K)
        
        corres = torch.einsum('nk,nkc->nc', attention_scores, knn_xyz.squeeze())  # (N, 3)
        dist = torch.norm(src_points - corres, dim=-1)
        weight = torch.relu(1. - dist / self.max_dist)    
        return corres, weight


class CompatibilityGraphEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, sigma_d):
        super(CompatibilityGraphEmbedding, self).__init__()
        self.num_layers = num_layers
        self.layer = nn.Linear(in_channels+6, out_channels)
        self.mlps = nn.ModuleList([UnaryBlock(out_channels, out_channels) for _ in range(num_layers)])
        self.attns = nn.ModuleList([AttentionLayer(out_channels, 1) for _ in range(num_layers)])
        self.classifier = nn.Sequential(
            nn.Linear(out_channels, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        self.sigma_spat = nn.Parameter(torch.tensor(sigma_d).float(), requires_grad=False)

    def forward(self, ref_keypts:torch.Tensor, src_keypts:torch.Tensor, corr_feat):
        feat = torch.cat([ref_keypts, src_keypts], dim=-1)  # (N, 6)
        feat = feat - feat.mean(-1, keepdim=True)  # (N, 6)
        feat = torch.cat([corr_feat, feat], dim=-1)  # (N, C+6)
        feat = self.layer(feat.unsqueeze(0))  # (1, N, C)

        with torch.no_grad():
            geo_compatibility = torch.cdist(ref_keypts, ref_keypts) - torch.cdist(src_keypts, src_keypts)
            geo_compatibility = torch.clamp(1.0 - geo_compatibility ** 2 / self.sigma_spat ** 2, min=0)
            geo_compatibility = geo_compatibility.unsqueeze(0)
        
        for i in range(self.num_layers):
            feat = self.mlps[i](feat)
            feat = self.attns[i](feat, feat, attention_factors=geo_compatibility)[0]
        
        feat = F.normalize(feat, p=2, dim=-1).squeeze(0)
        return feat, self.classifier(feat).squeeze()
