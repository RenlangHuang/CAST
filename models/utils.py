import torch
import numpy as np
from typing import Optional
import MinkowskiEngine as ME
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from pytorch3d.ops import ball_query, packed_to_padded


def weighted_svd(src_points: torch.Tensor, ref_points: torch.Tensor, weights: Optional[torch.Tensor]=None, orthogonalization=True):
    """Compute rigid transformation from `src_points` to `ref_points` using weighted SVD (Kabsch).

    Args:
        src_points: torch.Tensor (B, N, 3) or (N, 3)
        ref_points: torch.Tensor (B, N, 3) or (N, 3)
        weights: torch.Tensor (B, N) or (N,) (default: None)

    Returns:
    transform: torch.Tensor (B, 4, 4) or (4, 4)
    """
    if src_points.ndim == 2:
        src_points = src_points.unsqueeze(0)
        ref_points = ref_points.unsqueeze(0)
        if weights is not None:
            weights = weights.unsqueeze(0)
        squeeze_first = True
    else:
        squeeze_first = False

    batch_size = src_points.shape[0]
    if weights is None:
        weights = torch.ones_like(src_points[:, :, 0])
    else: weights = torch.clamp(weights, 0.)
    weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-5)
    weights = weights.unsqueeze(2)  # (B, N, 1)

    src_centroid = torch.sum(src_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
    ref_centroid = torch.sum(ref_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
    src_points_centered = src_points - src_centroid  # (B, N, 3)
    ref_points_centered = ref_points - ref_centroid  # (B, N, 3)

    H = src_points_centered.permute(0, 2, 1) @ (weights * ref_points_centered)
    U, _, V = torch.svd(H.cpu())  # H = USV^T, SVD operates faster on CPU than on GPU
    Ut, V = U.transpose(1, 2).to(H.device), V.to(H.device)
    eye = torch.eye(3, device=H.device).unsqueeze(0).repeat(batch_size, 1, 1)
    eye[:, -1, -1] = torch.sign(torch.det(V @ Ut))
    R = V @ eye @ Ut
    
    if orthogonalization:
        rot_0 = R[..., 0] / torch.norm(R[...,0], dim=-1, keepdim=True)
        rot_1 = R[..., 1] - torch.sum(R[..., 1] * rot_0, dim=-1, keepdim=True) * rot_0
        rot_1 = rot_1 / torch.norm(rot_1, dim=-1, keepdim=True)
        rot_2 = R[..., 2] - torch.sum(R[..., 2] * rot_0, dim=-1, keepdim=True) * rot_0 \
                          - torch.sum(R[..., 2] * rot_1, dim=-1, keepdim=True) * rot_1
        rot_2 = rot_2 / torch.norm(rot_2, dim=-1, keepdim=True)
        R = torch.stack([rot_0, rot_1, rot_2], dim=-1)

    t = ref_centroid.permute(0, 2, 1) - R @ src_centroid.permute(0, 2, 1)
    transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
    transform[:, :3, :3], transform[:, :3, 3] = R, t.squeeze(2)
    if squeeze_first: transform = transform.squeeze(0)
    return transform


def grid_subsample_gpu(points:torch.Tensor, batches_len:torch.Tensor, voxel_size):
    """
    Same as `grid_subsample`, but implemented in GPU using Minkowski engine's sparse quantization.
    Note: This function is not deterministic and may return subsampled points in a different order.
    """
    B = len(batches_len)
    batch_start_end = F.pad(torch.cumsum(batches_len, 0), (1, 0))
    device = points.device

    coord_batched = ME.utils.batched_coordinates(
        [points[batch_start_end[b]:batch_start_end[b + 1]] / voxel_size for b in range(B)], device=device)
    sparse_tensor = ME.SparseTensor(
        features=points,
        coordinates=coord_batched,
        quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE
    )
    s_points = sparse_tensor.features
    s_len = torch.tensor([f.shape[0] for f in sparse_tensor.decomposed_features], device=device)
    return s_points, s_len


def radius_search_gpu(queries, supports, q_batches, s_batches, radius, max_neighbors):
    """
    Same as `radius_search`, but implemented by GPU using PyTorch3D's ball_query functions.
    Computes neighbors for a batch of queries and supports, apply radius search
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B) the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """
    B = len(q_batches)
    N_spts_total = supports.shape[0]
    q_first_idx = F.pad(torch.cumsum(q_batches, dim=0)[:-1], (1, 0))
    queries_padded = packed_to_padded(queries, q_first_idx, q_batches.max().item())  # (B, N_max, 3)
    s_first_idx = F.pad(torch.cumsum(s_batches, dim=0)[:-1], (1, 0))
    supports_padded = packed_to_padded(supports, s_first_idx, s_batches.max().item())  # (B, N_max, 3)

    idx = ball_query(queries_padded, supports_padded,
                     q_batches, s_batches,
                     K=max_neighbors, radius=radius).idx  # (N_clouds, N_pts, K)
    idx[idx < 0] = torch.iinfo(idx.dtype).min

    idx_packed = torch.cat([idx[b][:q_batches[b]] + s_first_idx[b] for b in range(B)], dim=0)
    idx_packed[idx_packed < 0] = N_spts_total

    return idx_packed



def index_select(data: torch.Tensor, index: torch.LongTensor, dim: int) -> torch.Tensor:
    """Advanced index select.

    Returns a tensor `output` which indexes the `data` tensor along dimension `dim`
    using the entries in `index` which is a `LongTensor`.

    Different from `torch.index_select`, `index` does not has to be 1-D. The `dim`-th
    dimension of `data` will be expanded to the number of dimensions in `index`.

    For example, suppose the shape `data` is $(a_0, a_1, ..., a_{n-1})$, the shape of `index` is
    $(b_0, b_1, ..., b_{m-1})$, and `dim` is $i$, then `output` is $(n+m-1)$-d tensor, whose shape is
    $(a_0, ..., a_{i-1}, b_0, b_1, ..., b_{m-1}, a_{i+1}, ..., a_{n-1})$.

    Args:
        data (Tensor): (a_0, a_1, ..., a_{n-1})
        index (LongTensor): (b_0, b_1, ..., b_{m-1})
        dim: int

    Returns:
        output (Tensor): (a_0, ..., a_{dim-1}, b_0, ..., b_{m-1}, a_{dim+1}, ..., a_{n-1})
    """
    output = data.index_select(dim, index.view(-1))
    if index.ndim > 1:
        output_shape = data.shape[:dim] + index.shape + data.shape[dim:][1:]
        output = output.view(*output_shape)
    return output


def pairwise_distance(x: torch.Tensor, y: torch.Tensor, normalized = False) -> torch.Tensor:
    """Pairwise distance of two (batched) point clouds.

    Args:
        x (Tensor): (*, N, C) or (*, C, N)
        y (Tensor): (*, M, C) or (*, C, M)
        normalized (bool=False): if the points are normalized, "x2 + y2 = 1", so "d2 = 2 - 2xy".

    Returns:
        dist: torch.Tensor (*, N, M)
    """
    xy = torch.matmul(x, y.transpose(-1, -2))  # (*, N, C) x [(*, M, C) -> (*, C, M)]
    if normalized:
        sq_distances = 2.0 - 2.0 * xy
    else:
        x2 = torch.sum(x ** 2, dim=-1).unsqueeze(-1)  # (*, N, C) or (*, C, N) -> (*, N) -> (*, N, 1)
        y2 = torch.sum(y ** 2, dim=-1).unsqueeze(-2)  # (*, M, C) or (*, C, M) -> (*, M) -> (*, 1, M)
        sq_distances = x2 - 2.0 * xy + y2
    return sq_distances.clamp(min=0.0)


def apply_transform(points: torch.Tensor, transform: torch.Tensor, normals: Optional[torch.Tensor] = None):
    """Rigid transform to points and normals (optional). There are two cases supported:
    1. points and normals are (*, 3), transform is (4, 4), the output points are (*, 3).
       In this case, the transform is applied to all points.
    2. points and normals are (B, N, 3), transform is (B, 4, 4), the output points are (B, N, 3).
       In this case, the transform is applied batch-wise. The points can be broadcast if B=1.

    Args:
        points (Tensor): (*, 3) or (B, N, 3)
        normals (optional[Tensor]=None): same shape as points.
        transform (Tensor): (4, 4) or (B, 4, 4)

    Returns:
        points (Tensor): same shape as points.
        normals (Tensor): same shape as points.
    """
    if normals is not None:
        assert points.shape == normals.shape
    if transform.ndim == 2:
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        points_shape = points.shape
        points = points.reshape(-1, 3)
        points = torch.matmul(points, rotation.transpose(-1, -2)) + translation
        points = points.reshape(*points_shape)
        if normals is not None:
            normals = normals.reshape(-1, 3)
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
            normals = normals.reshape(*points_shape)
    elif transform.ndim == 3 and points.ndim == 3:
        rotation = transform[:, :3, :3]  # (B, 3, 3)
        translation = transform[:, None, :3, 3]  # (B, 1, 3)
        points = torch.matmul(points, rotation.transpose(-1, -2)) + translation
        if normals is not None:
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
    else: raise ValueError('Incompatible tensor shapes.')
    if normals is not None:
        return points, normals
    else:
        return points


@torch.no_grad()
def point_to_node_partition(points: torch.Tensor, nodes: torch.Tensor, point_limit):
    """Point-to-Node partition to the point cloud.

    Args:
        points (Tensor): (N, 3)
        nodes (Tensor): (M, 3)
        point_limit (int): max number of points to each node

    Returns:
        node_masks (BoolTensor): (M,)
        node_knn_indices (LongTensor): (M, K)
        node_knn_masks (BoolTensor) (M, K)
    """
    sq_dist_mat = pairwise_distance(nodes, points)  # (M, N)
    point_to_node = sq_dist_mat.min(dim=0)[1]  # (N,)
    node_masks = torch.zeros_like(nodes[:,0], dtype=torch.bool)  # (M,)
    node_masks.index_fill_(0, point_to_node, True)

    matching_masks = torch.zeros_like(sq_dist_mat, dtype=torch.bool)  # (M, N)
    point_indices = torch.arange(points.shape[0], device=points.device)  # (N,)
    matching_masks[point_to_node, point_indices] = True  # (M, N)
    sq_dist_mat.masked_fill_(~matching_masks, 1e12)  # (M, N)

    node_knn_indices = sq_dist_mat.topk(k=point_limit, dim=1, largest=False)[1]  # (M, K)
    node_knn_node_indices = index_select(point_to_node, node_knn_indices, dim=0)  # (M, K)
    node_indices = torch.arange(nodes.shape[0], device=nodes.device).unsqueeze(1)  # (M, 1)
    node_knn_masks = node_knn_node_indices.eq(node_indices)  # (M, K)
    node_knn_indices.masked_fill_(~node_knn_masks, points.shape[0])

    return node_masks, node_knn_indices, node_knn_masks


def generate_rand_rotm(x_lim=5.0, y_lim=5.0, z_lim=180.0) -> np.ndarray:
    rand_z = np.random.uniform(low=-z_lim, high=z_lim)
    rand_y = np.random.uniform(low=-y_lim, high=y_lim)
    rand_x = np.random.uniform(low=-x_lim, high=x_lim)

    rand_eul = np.array([rand_z, rand_y, rand_x])
    r = Rotation.from_euler('zyx', rand_eul, degrees=True)
    rotm = r.as_matrix()
    return rotm
