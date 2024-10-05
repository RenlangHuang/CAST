import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points
from models.utils import index_select
from models.kpconv.kpconv import KPConv


def nearest_upsample(x, upsample_indices):
    """Pools features from the closest neighbors.

    Args:
        x: [M, C] features matrix
        upsample_indices: [N, K] Only the first column is used for pooling

    Returns:
        x: [N, C] pooled features matrix
    """
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)
    x = index_select(x, upsample_indices[:, 0], dim=0)
    return x


def maxpool(x, neighbor_indices):
    """Max pooling from neighbors.

    Args:
        x: [M, C] features matrix
        neighbor_indices: [N, K] pooling indices

    Returns:
        pooled_feats: [N, C] pooled features matrix
    """
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)
    neighbor_feats = index_select(x, neighbor_indices, dim=0)
    pooled_feats = neighbor_feats.max(1)[0]
    return pooled_feats


class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels):
        super(GroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.norm = nn.GroupNorm(self.num_groups, self.num_channels)

    def forward(self, x: torch.Tensor):
        ndim = x.ndim
        if ndim == 2:
            x = x.unsqueeze(0)  # (N, C) -> (1, N, C)
        x = x.transpose(1, 2)  # (B, N, C) -> (B, C, N)
        x = self.norm(x)
        x = x.transpose(1, 2)  # (B, C, N) -> (B, N, C)
        if ndim == 2:
            x = x.squeeze(0)
        return x
    
    def __repr__(self):
        return self.norm.__repr__()


ACT_LAYERS = {
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU(0.1),
    'sigmoid': nn.Sigmoid(),
    'softplus': nn.Softplus(),
    'tanh': nn.Tanh(),
    'elu': nn.ELU(),
    'gelu': nn.GELU(),
    None: nn.Identity(),
}

class UnaryBlock(nn.Module):
    def __init__(self, in_channels, out_channels, group_norm=32, activation_fn='leaky_relu', bias=True, layer_norm=False):
        super(UnaryBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation_fn = activation_fn
        self.mlp = nn.Linear(in_channels, out_channels, bias=bias)
        if layer_norm:
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = GroupNorm(group_norm, out_channels)
        self.activation = ACT_LAYERS[activation_fn]

    def forward(self, x):
        x = self.mlp(x)
        x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '{}, {}'.format(self.in_channels, self.out_channels)
        format_string += ', ' + self.norm.__repr__()
        format_string += ', ' + self.activation.__repr__()
        format_string += ')'
        return format_string


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, radius,
                 sigma, group_norm=32, bias=True, layer_norm=False):
        super(ConvBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.KPConv = KPConv(in_channels, out_channels, kernel_size, radius, sigma, bias)
        if layer_norm:
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = GroupNorm(group_norm, out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, s_feats, q_points, s_points, neighbor_indices):
        x = self.KPConv(s_feats, q_points, s_points, neighbor_indices)
        return self.leaky_relu(self.norm(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, radius,
        sigma, group_norm=32, strided=False, bias=True, layer_norm=False,
    ):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strided = strided
        mid_channels = out_channels // 4

        if in_channels != mid_channels:
            self.unary1 = UnaryBlock(in_channels, mid_channels, group_norm, bias=bias, layer_norm=layer_norm)
        else:
            self.unary1 = nn.Identity()

        self.KPConv = KPConv(mid_channels, mid_channels, kernel_size, radius, sigma, bias=bias)
        if layer_norm:
            self.norm = nn.LayerNorm(mid_channels)
        else:
            self.norm = GroupNorm(group_norm, mid_channels)

        self.unary2 = UnaryBlock(
            mid_channels, out_channels, group_norm, activation_fn=None, bias=bias, layer_norm=layer_norm
        )

        if in_channels != out_channels:
            self.unary_shortcut = UnaryBlock(
                in_channels, out_channels, group_norm, activation_fn=None, bias=bias, layer_norm=layer_norm
            )
        else:
            self.unary_shortcut = nn.Identity()

        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, s_feats, q_points, s_points, neighbor_indices):
        x = self.unary1(s_feats)
        x = self.KPConv(x, q_points, s_points, neighbor_indices)
        x = self.leaky_relu(self.norm(x))
        x = self.unary2(x)
        if self.strided:
            shortcut = maxpool(s_feats, neighbor_indices)
        else:
            shortcut = s_feats
        shortcut = self.unary_shortcut(shortcut)
        return self.leaky_relu(x + shortcut)


class NearestUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, group_norm=32):
        super(NearestUpsampleBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group_norm = group_norm
        if isinstance(self.group_norm, int):
            self.unary = UnaryBlock(in_channels, out_channels, group_norm)
        else:
            self.unary = nn.Linear(in_channels, out_channels)
    
    def forward(self, query, support, upsample_indices):
        latent = nearest_upsample(support, upsample_indices)
        latent = torch.cat([latent, query], dim=1)
        return self.unary(latent)


def knn_group(xyz1: torch.Tensor, xyz2: torch.Tensor, features2: torch.Tensor, k):
    _, knn_idx, knn_xyz = knn_points(xyz1.unsqueeze(0), xyz2.unsqueeze(0), K=k, return_nn=True)
    knn_idx, knn_xyz = torch.squeeze(knn_idx, dim=0), torch.squeeze(knn_xyz, dim=0)
    rela_xyz = knn_xyz - xyz1.unsqueeze(1)  # (M, k, 3)
    rela_dist = torch.norm(rela_xyz, dim=-1, keepdim=True)  # (M, k, 1)
    grouped_features =  torch.cat([rela_xyz, rela_dist], dim=-1)  # (M, k, 4)
    if features2 is not None:
        knn_features = index_select(features2, knn_idx, dim=0)  # (M, k, C)
        grouped_features = torch.cat([rela_xyz, rela_dist, knn_features], dim=-1)
    return grouped_features, knn_xyz


class KeypointDetector(nn.Module):
    def __init__(self, k, in_channels, out_channels):
        super(KeypointDetector, self).__init__()
        self.k = k
        self.convs = nn.Sequential(
            UnaryBlock(in_channels+4, out_channels, bias=False),
            UnaryBlock(out_channels, out_channels, bias=False)
        )
        self.mlp = nn.Sequential(
            UnaryBlock(out_channels, out_channels),
            nn.Linear(out_channels, 1),
            nn.Softplus()
        )
    
    def forward(self, sampled_xyz, xyz, features):
        grouped_features, knn_xyz = knn_group(sampled_xyz, xyz, features, self.k)
        embedding: torch.Tensor = self.convs(grouped_features)  # (M, k, C)
        attentive_weights = F.softmax(embedding.max(dim=-1)[0], dim=-1).unsqueeze(-1)  # (M, k, 1)
        keypoints = torch.sum(attentive_weights * knn_xyz, dim=-2)  # (M, k, 3)

        attentive_feature_map = embedding * attentive_weights  # (M, k, C)
        attentive_feature = torch.sum(attentive_feature_map, dim=-2)  # (M, k, C)
        sigmas = torch.squeeze(self.mlp(attentive_feature) + 0.001, dim=-1) # (M,)
        return keypoints, sigmas, grouped_features[..., 4:], attentive_feature_map


class DescExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DescExtractor, self).__init__()
        self.mlp = nn.Sequential(
            UnaryBlock(in_channels*2+out_channels, out_channels, bias=False),
            UnaryBlock(out_channels, out_channels, bias=False),
        )
    
    def forward(self, x1, attentive_feature_map):
        #x1 = self.convs(grouped_features), # (B, N, k, C)
        x2 = torch.max(x1, dim=1, keepdim=True)[0] # (N, 1, C_in)
        x2 = x2.repeat(1, x1.shape[1], 1) # (N, k, C_in)
        x2 = torch.cat((x2, x1), dim=-1) # (N, k, 2C_in)
        x2 = torch.cat((x2, attentive_feature_map), dim=-1) # (N, k, 2C_in+C_det)
        desc = torch.max(self.mlp(x2), dim=1, keepdim=False)[0] # (N, C_out)
        return torch.relu(desc)