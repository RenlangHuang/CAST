"""
Ablation study 4: Full version of CAST
"""

import torch
import torch.nn as nn
from pytorch3d.ops import knn_points, knn_gather

from models.kpconv import KPConvFPN, UnaryBlock
from models.cast.cast import SpotGuidedGeoTransformer
from models.cast.correspondence import KeypointMatching, FineMatching, CompatibilityGraphEmbedding
from models.utils import grid_subsample_gpu, radius_search_gpu, apply_transform, weighted_svd
from models.cast.consistency import registration_ransac_based_on_correspondence


class CAST(nn.Module):
    def __init__(self, cfg):
        super(CAST, self).__init__()
        self.kpconv_layers = cfg.kpconv_layers
        self.voxel_size = cfg.voxel_size
        self.init_radius = cfg.init_radius
        self.neighbor_limits = cfg.neighbor_limits

        self.sigma_r = cfg.sigma_r
        self.patch_k = cfg.patch_k
        self.use_overlap_head = cfg.use_overlap_head
        self.overlap_threshold = cfg.overlap_threshold
        self.keypoint_node_threshold = cfg.keypoint_node_threshold
        self.local_matching_radius = cfg.local_matching_radius
        self.ransac = cfg.ransac
        if cfg.ransac:
            self.ransac_filter = cfg.ransac_filter

        self.backbone = KPConvFPN(cfg)
        self.transformer = SpotGuidedGeoTransformer(cfg)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.output_dim)
        if cfg.use_overlap_head:
            self.overlap_head = nn.Sequential(
                UnaryBlock(cfg.hidden_dim + cfg.blocks, cfg.hidden_dim),
                UnaryBlock(cfg.hidden_dim, cfg.hidden_dim),
                nn.Linear(cfg.hidden_dim, 1), nn.Sigmoid()
            )

        self.keypoint_matching = KeypointMatching(cfg.init_dim, cfg.num_neighbors, cfg.learnable_matcher)
        self.corres_embedding = UnaryBlock(cfg.init_dim*2 + 3, cfg.init_dim)
        self.consistency_filter = CompatibilityGraphEmbedding(
            cfg.init_dim, cfg.init_dim, cfg.filter_layers, cfg.filter_sigma_d
        )
        self.descriptor = nn.Linear(cfg.init_dim * 4, cfg.desc_dim)
        self.fine_matching = FineMatching(cfg.desc_dim, cfg.dense_neighbors, cfg.local_matching_radius)
    
    @torch.no_grad()
    def preprocess(self, points):
        points_list, length_list = [points], []
        lengths = torch.LongTensor([points.shape[0]]).to(points.device)
        length_list.append(lengths)

        voxel_size = self.voxel_size
        radius = self.init_radius

        for _ in range(self.kpconv_layers - 1):
            voxel_size = voxel_size * 2.
            points, lengths = grid_subsample_gpu(points, lengths, voxel_size)
            points_list.append(points); length_list.append(lengths)
        
        neighbors_list = []
        subsampling_list = []
        upsampling_list = [None]
        
        for i in range(self.kpconv_layers):
            neighbors_list.append(radius_search_gpu(
                points_list[i], points_list[i], length_list[i], length_list[i], radius, self.neighbor_limits[i]
            ))
            if i == self.kpconv_layers - 1: break
            subsampling_list.append(radius_search_gpu(
                points_list[i + 1], points_list[i], length_list[i + 1], length_list[i], radius, self.neighbor_limits[i]
            ))
            radius = radius * 2.
            if i == 0: continue
            upsampling_list.append(torch.squeeze(knn_points(
                points_list[i].unsqueeze(0), points_list[i + 1].unsqueeze(0))[1], dim=0)
            )
        return points_list, neighbors_list, subsampling_list, upsampling_list
    
    def forward(self, ref_points, src_points, gt_transform, use_gt_mask=True):
        # 1. Preprocess the original point clouds
        points_list1, neighbors_list1, subsampling_list1, upsampling_list1 = self.preprocess(ref_points[0])
        points_list2, neighbors_list2, subsampling_list2, upsampling_list2 = self.preprocess(src_points[0])

        # 2. Extract hierarchical feature maps and sparse keypoints
        ref_dict = self.backbone(points_list1, neighbors_list1, subsampling_list1, upsampling_list1)
        src_dict = self.backbone(points_list2, neighbors_list2, subsampling_list2, upsampling_list2)
        
        output_dict = {
            'gt_transform': gt_transform[0],
            'ref_kpts': ref_dict['keypoints'],
            'ref_sigma': ref_dict['sigma'],
            'src_kpts': src_dict['keypoints'],
            'src_sigma': src_dict['sigma'],
        }
        
        # 3. Interaction of coarse voxelized features
        ref_feats, src_feats, correlation, ref_compat, src_compat = self.transformer(
            points_list1[2].unsqueeze(0),
            points_list2[2].unsqueeze(0),
            ref_dict['feats'][1].unsqueeze(0),
            src_dict['feats'][1].unsqueeze(0),
            points_list1[-1].unsqueeze(0),
            points_list2[-1].unsqueeze(0),
            ref_dict['feats'][-1].unsqueeze(0),
            src_dict['feats'][-1].unsqueeze(0),
        )
        if self.use_overlap_head:
            ref_patch_overlap = torch.cat([ref_feats, ref_compat], dim=-1)
            ref_patch_overlap = self.overlap_head(ref_patch_overlap).squeeze(0) # (M, 1)
            src_patch_overlap = torch.cat([src_feats, src_compat], dim=-1)
            src_patch_overlap = self.overlap_head(src_patch_overlap).squeeze(0) # (N, 1)
            output_dict['ref_patch_overlap'] = ref_patch_overlap.squeeze(-1)
            output_dict['src_patch_overlap'] = src_patch_overlap.squeeze(-1)

        ref_feats = self.out_proj(ref_feats).squeeze(0) # (M, C)
        src_feats = self.out_proj(src_feats).squeeze(0) # (N, C)
        
        output_dict['ref_feats_c'] = ref_feats
        output_dict['src_feats_c'] = src_feats
        output_dict['spot_matching_scores'] = torch.cat(correlation)

        with torch.no_grad():
            # 4. Generate ground-truth patch correspondences
            dist = torch.cdist(points_list1[2], apply_transform(points_list2[2], gt_transform[0]))
            dist = torch.clamp_max(dist / self.sigma_r, 2.)
            overlap = torch.relu(1. + dist.pow(3) / 16. - 0.75 * dist)
            
            gt_patch_corr_indices = overlap.nonzero()
            gt_patch_corr_overlaps = overlap[gt_patch_corr_indices[:, 0], gt_patch_corr_indices[:, 1]]
            output_dict['gt_patch_corr_indices'] = gt_patch_corr_indices
            output_dict['gt_patch_corr_overlaps'] = gt_patch_corr_overlaps
        
        # 5. Generate patch correspondences via coarse matching
        matching_scores = self.transformer.matching_scores(ref_feats, src_feats)
        if self.use_overlap_head:
            matching_scores = matching_scores * ref_patch_overlap * src_patch_overlap.transpose(-1, -2)
        with torch.no_grad():
            matching_mask = torch.zeros_like(matching_scores, dtype=torch.bool)
            matching_mask = torch.logical_and(
                matching_mask.scatter(-1, matching_scores.max(-1,True)[1], 1),
                matching_mask.scatter(-2, matching_scores.max(-2,True)[1], 1)
            )
            ref_patch_corr_indices, src_patch_corr_indices = matching_mask.nonzero(as_tuple=True)
            valid_ref_patch_node = torch.index_select(points_list1[2], 0, ref_patch_corr_indices)
            valid_src_patch_node = torch.index_select(points_list2[2], 0, src_patch_corr_indices)
        
        output_dict['ref_patch_corr_indices'] = ref_patch_corr_indices
        output_dict['src_patch_corr_indices'] = src_patch_corr_indices
        output_dict['coarse_matching_scores'] = matching_scores

        if self.training and use_gt_mask:
            with torch.no_grad():
                matching_scores_mask = torch.zeros_like(matching_scores, dtype=torch.bool)
                gt_patch_corr_indices = gt_patch_corr_indices[gt_patch_corr_overlaps.gt(self.overlap_threshold)]
                matching_scores_mask[gt_patch_corr_indices[:, 0], gt_patch_corr_indices[:, 1]] = True
                matching_scores.masked_fill_(~matching_scores_mask, 0.)

                matching_mask = torch.zeros_like(matching_scores, dtype=torch.bool)
                matching_mask = torch.logical_and(
                    matching_mask.scatter(-1, matching_scores.max(-1,True)[1], 1),
                    matching_mask.scatter(-2, matching_scores.max(-2,True)[1], 1)
                )
                ref_patch_corr_indices, src_patch_corr_indices = matching_mask.nonzero(as_tuple=True)
                valid_ref_patch_node = torch.index_select(points_list1[2], 0, ref_patch_corr_indices)
                valid_src_patch_node = torch.index_select(points_list2[2], 0, src_patch_corr_indices)

        # 6. Generate keypoint-to-patch correspondences
        with torch.no_grad():
            kpt_patch_dist, kpt_patch_index, _ = knn_points(ref_dict['keypoints'][None], valid_ref_patch_node[None])
            #patch_knn = knn_points(valid_src_patch_node[None], src_dict['keypoints'][None], K=self.patch_k)[1]
            patch_knn_mask, patch_knn, _ = knn_points(valid_src_patch_node[None], src_dict['keypoints'][None], K=self.patch_k)
            patch_knn_mask = torch.lt(patch_knn_mask, self.sigma_r).squeeze(0)  # (M, K)
            ref_kpt_mask = torch.squeeze(kpt_patch_dist).lt(self.keypoint_node_threshold).nonzero().squeeze(-1)
            ref_corr_index = torch.squeeze(kpt_patch_index).index_select(0, ref_kpt_mask)
        
        src_patch_knn_kpts = knn_gather(src_dict['keypoints'].unsqueeze(0), patch_knn).squeeze(0)  # (M, K, 3)
        src_patch_knn_desc = knn_gather(src_dict['desc'].unsqueeze(0), patch_knn).squeeze(0)  # (M, K, C)
        src_patch_knn_sigma = knn_gather(src_dict['sigma'].view(1, -1, 1), patch_knn).squeeze(0)  # (M, K, 1)
        
        src_patch_knn_kpts = src_patch_knn_kpts.index_select(0, ref_corr_index)
        src_patch_knn_desc = src_patch_knn_desc.index_select(0, ref_corr_index)
        src_patch_knn_sigma = src_patch_knn_sigma.index_select(0, ref_corr_index)
        patch_knn_mask = patch_knn_mask.index_select(0, ref_corr_index)
        output_dict['src_patch_corr_kpts'] = src_patch_knn_kpts

        # 7. Correspondence prediction and pose estimation
        src_corres_xyz, corr_feat, match_logits = self.keypoint_matching(
            ref_dict['desc'].index_select(0, ref_kpt_mask),
            src_patch_knn_kpts, src_patch_knn_desc,
            ref_dict['sigma'].index_select(0, ref_kpt_mask), src_patch_knn_sigma#, patch_knn_mask
        )
        ref_corres_xyz = ref_dict['keypoints'].index_select(0, ref_kpt_mask)
        corres_xyz = torch.cat([ref_corres_xyz, src_corres_xyz], dim=1)
        corr_feat = self.corres_embedding(corr_feat)
        
        corr_feat, corr_weight = self.consistency_filter(ref_corres_xyz.detach(), src_corres_xyz.detach(), corr_feat)

        if not self.training and self.ransac:
            transform = registration_ransac_based_on_correspondence(
                valid_ref_patch_node, valid_src_patch_node,
                matching_scores[ref_patch_corr_indices, src_patch_corr_indices],
            )
            #transform = registration_ransac_based_on_correspondence(ref_corres_xyz, src_corres_xyz, corr_weight)
            mask = torch.norm(apply_transform(src_corres_xyz, transform) - ref_corres_xyz, dim=-1)
            corr_weight = corr_weight * torch.lt(mask, self.ransac_filter).float()
        
        transform = weighted_svd(src_corres_xyz, ref_corres_xyz, corr_weight, not self.training)

        output_dict['corres'] = corres_xyz
        output_dict['match_logits'] = match_logits
        output_dict['corr_confidence'] = corr_weight
        output_dict['transform'] = transform

        # 8. Fine matching and pose refinement
        src_corres_xyz = apply_transform(src_corres_xyz, transform)
        dist = torch.norm(src_corres_xyz - ref_corres_xyz, dim=-1)
        corr_weight = corr_weight * torch.lt(dist, self.local_matching_radius).float()
        
        ref_desc = self.descriptor(ref_dict['feats'][0])
        src_desc = self.descriptor(src_dict['feats'][0])
        src_xyz_f = apply_transform(points_list2[1], transform)
        ref_xyz_f, weight = self.fine_matching(points_list1[1], src_xyz_f, ref_desc, src_desc)

        dist = torch.norm(src_xyz_f - ref_xyz_f, dim=-1)
        src_corres_xyz = torch.cat([src_corres_xyz, src_xyz_f], dim=0)
        ref_corres_xyz = torch.cat([ref_corres_xyz, ref_xyz_f], dim=0)
        corr_weight = torch.cat([corr_weight, weight], dim=0)
        refined_transform = weighted_svd(src_corres_xyz, ref_corres_xyz, corr_weight, not self.training)

        output_dict['refined_transform'] = refined_transform @ transform

        return output_dict