import torch
import torch.nn as nn
from typing import Dict
import torch.nn.functional as F
from models.utils import pairwise_distance, apply_transform


class ProbChamferLoss(nn.Module):
    def __init__(self):
          super(ProbChamferLoss, self).__init__()
    
    def forward(self, output_dict: Dict[str, torch.Tensor]):
        keypoints1 = output_dict['ref_kpts']
        keypoints2 = apply_transform(output_dict['src_kpts'], output_dict['gt_transform'])
        diff = torch.norm(keypoints1.unsqueeze(1) - keypoints2.unsqueeze(0), dim=-1)

        if output_dict['ref_sigma'] is None or output_dict['src_sigma'] is None:
            min_dist_forward, _ = torch.min(diff, dim=-1)
            forward_loss = min_dist_forward.mean()
            min_dist_backward, _ = torch.min(diff, dim=-2)
            backward_loss = min_dist_backward.mean()
        else:
            min_dist_forward, min_dist_forward_I = torch.min(diff, dim=-1)
            selected_sigma_2 = output_dict['src_sigma'].index_select(0, min_dist_forward_I)
            sigma_forward = (output_dict['ref_sigma'] + selected_sigma_2) / 2.
            forward_loss = (sigma_forward.log() + min_dist_forward / sigma_forward).mean()

            min_dist_backward, min_dist_backward_I = torch.min(diff, dim=-2)
            selected_sigma_1 = output_dict['ref_sigma'].index_select(0, min_dist_backward_I)
            sigma_backward = (output_dict['src_sigma'] + selected_sigma_1) / 2.
            backward_loss = (sigma_backward.log() + min_dist_backward / sigma_backward).mean()

        return forward_loss + backward_loss


class WeightedCircleLoss(nn.Module):
    def __init__(self, pos_margin, neg_margin, pos_optimal, neg_optimal, log_scale, bilateral=True):
        super(WeightedCircleLoss, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.pos_optimal = pos_optimal
        self.neg_optimal = neg_optimal
        self.log_scale = log_scale
        self.bilateral = bilateral

    def forward(self, pos_masks:torch.Tensor, neg_masks:torch.Tensor, feat_dists, pos_scales=None, neg_scales=None):
        with torch.no_grad():
            row_masks = (torch.gt(pos_masks.sum(-1), 0) & torch.gt(neg_masks.sum(-1), 0)).nonzero().squeeze()
            if self.bilateral:
                col_masks = (torch.gt(pos_masks.sum(-2), 0) & torch.gt(neg_masks.sum(-2), 0)).nonzero().squeeze()
            
            pos_weights = torch.relu(feat_dists - 1e5 * (~pos_masks).float() - self.pos_optimal)
            neg_weights = torch.relu(self.neg_optimal - feat_dists - 1e5 * (~neg_masks).float())
            if pos_scales is not None: pos_weights = pos_weights * pos_scales
            if neg_scales is not None: neg_weights = neg_weights * neg_scales

        loss_pos_row = torch.logsumexp(self.log_scale * (feat_dists - self.pos_margin) * pos_weights, dim=-1)
        loss_neg_row = torch.logsumexp(self.log_scale * (self.neg_margin - feat_dists) * neg_weights, dim=-1)
        loss_row = F.softplus(loss_pos_row + loss_neg_row) / self.log_scale
        loss_row = torch.index_select(loss_row, 0, row_masks)
        
        if not self.bilateral:
            return loss_row.mean()
        
        loss_pos_col = torch.logsumexp(self.log_scale * (feat_dists - self.pos_margin) * pos_weights, dim=-2)
        loss_neg_col = torch.logsumexp(self.log_scale * (self.neg_margin - feat_dists) * neg_weights, dim=-2)
        loss_col = F.softplus(loss_pos_col + loss_neg_col) / self.log_scale
        loss_col = torch.index_select(loss_col, 0, col_masks)
        
        return (loss_row.mean() + loss_col.mean()) / 2.


class SpotMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(SpotMatchingLoss, self).__init__()
        self.positive_overlap = cfg.positive_overlap
    
    def forward(self, output_dict):
        coarse_matching_scores = output_dict['coarse_matching_scores']
        gt_node_corr_indices = output_dict['gt_patch_corr_indices']
        gt_node_corr_overlaps = output_dict['gt_patch_corr_overlaps']
        
        with torch.no_grad():
            overlaps = torch.zeros_like(coarse_matching_scores)
            overlaps[gt_node_corr_indices[:, 0], gt_node_corr_indices[:, 1]] = gt_node_corr_overlaps
            pos_masks = torch.gt(overlaps, self.positive_overlap)

            row_mask = torch.zeros_like(overlaps, dtype=torch.bool)
            idx = overlaps.max(dim=1, keepdim=True)[1]
            row_mask.scatter_(1, idx, True)
            col_mask = torch.zeros_like(overlaps, dtype=torch.bool)
            idx = overlaps.max(dim=0, keepdim=True)[1]
            col_mask.scatter_(0, idx, True)
            pos_masks = overlaps * (pos_masks & row_mask & col_mask).float()
        
        if 'spot_matching_scores' in output_dict.keys():
            matching_scores = output_dict['spot_matching_scores']
            loss = -torch.log(matching_scores + 1e-8) * pos_masks.unsqueeze(0)
            loss = torch.sum(loss) / pos_masks.sum() / matching_scores.shape[0]
        
        coarse_loss = -torch.log(coarse_matching_scores + 1e-8) * pos_masks
        coarse_loss = torch.sum(coarse_loss) / pos_masks.sum()

        if 'ref_patch_overlap' in output_dict.keys():
            gt_ref_patch_overlap = 1. - pos_masks.sum(-1).gt(0).float()
            gt_src_patch_overlap = 1. - pos_masks.sum(-2).gt(0).float()
            gt_ref_patch_overlap = gt_ref_patch_overlap / (gt_ref_patch_overlap.sum() + 1e-8)
            gt_src_patch_overlap = gt_src_patch_overlap / (gt_src_patch_overlap.sum() + 1e-8)
            loss_ref_ov = -torch.log(1. - output_dict['ref_patch_overlap'] + 1e-8) * gt_ref_patch_overlap
            loss_src_ov = -torch.log(1. - output_dict['src_patch_overlap'] + 1e-8) * gt_src_patch_overlap
            #coarse_loss = coarse_loss + loss_ref_ov.mean() + loss_src_ov.mean()
            coarse_loss = coarse_loss + loss_ref_ov.sum() + loss_src_ov.sum()
            #loss = loss + loss_ref_ov.mean() + loss_src_ov.mean()
        
        if 'spot_matching_scores' in output_dict.keys():
            return loss, coarse_loss
        else: return coarse_loss


class CoarseMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(CoarseMatchingLoss, self).__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            cfg.positive_margin,
            cfg.negative_margin,
            cfg.positive_optimal,
            cfg.negative_optimal,
            cfg.log_scale,
        )
        self.positive_overlap = cfg.positive_overlap
    
    def forward(self, output_dict):
        ref_feats = output_dict['ref_feats_c']
        src_feats = output_dict['src_feats_c']
        gt_node_corr_indices = output_dict['gt_patch_corr_indices']
        gt_node_corr_overlaps = output_dict['gt_patch_corr_overlaps']

        feat_dists = torch.sqrt(pairwise_distance(ref_feats, src_feats, normalized=True))
        overlaps = torch.zeros_like(feat_dists)
        overlaps[gt_node_corr_indices[:, 0], gt_node_corr_indices[:, 1]] = gt_node_corr_overlaps
        pos_masks = torch.gt(overlaps, self.positive_overlap)
        neg_masks = torch.eq(overlaps, 0)
        pos_scales = torch.sqrt(overlaps * pos_masks.float())
        
        return self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)


class CorrespondenceLoss(nn.Module):
    def __init__(self, point_to_patch_threshold):
          super(CorrespondenceLoss, self).__init__()
          self.point_to_patch_threshold = point_to_patch_threshold
    
    def forward(self, output_dict: Dict[str, torch.Tensor]):
        gt_transform = output_dict['gt_transform']
        with torch.no_grad():
            ref_kpts = apply_transform(output_dict['ref_kpts'], torch.linalg.inv(gt_transform))
            dist = torch.norm(output_dict['src_patch_corr_kpts'] - ref_kpts.unsqueeze(1), dim=-1)
            ref_mask = torch.lt(dist.min(dim=-1)[0], self.point_to_patch_threshold).nonzero().squeeze()

            src_kpts = apply_transform(output_dict['src_kpts'], gt_transform)
            dist = torch.norm(output_dict['ref_patch_corr_kpts'] - src_kpts.unsqueeze(1), dim=-1)
            src_mask = torch.lt(dist.min(dim=-1)[0], self.point_to_patch_threshold).nonzero().squeeze()

        loss_corr_ref = torch.norm(output_dict['ref_corres'] - ref_kpts, dim=-1)
        loss_corr_src = torch.norm(output_dict['src_corres'] - src_kpts, dim=-1)

        loss_corr_ref = torch.index_select(loss_corr_ref, 0, ref_mask)
        loss_corr_src = torch.index_select(loss_corr_src, 0, src_mask)
        return (loss_corr_ref.mean() + loss_corr_src.mean()) / 2.


class KeypointMatchingLoss(nn.Module):
    """
    Modified from source codes of:
     - REGTR https://github.com/yewzijian/RegTR.
    """
    def __init__(self, positive_threshold, negative_threshold):
        super(KeypointMatchingLoss, self).__init__()
        self.r_p = positive_threshold
        self.r_n = negative_threshold
    
    def cal_loss(self, src_xyz, tgt_grouped_xyz, tgt_corres, match_logits, match_score, transform):
        tgt_grouped_xyz = apply_transform(tgt_grouped_xyz, transform)
        tgt_corres = apply_transform(tgt_corres, transform)

        with torch.no_grad():
            dist_keypts:torch.Tensor = torch.norm(src_xyz.unsqueeze(1) - tgt_grouped_xyz, dim=-1)
            dist1, idx1 = torch.topk(dist_keypts, k=1, dim=-1, largest=False)
            mask = dist1[..., 0] < self.r_p  # Only consider points with correspondences
            ignore = dist_keypts < self.r_n  # Ignore all the points within a certain boundary
            ignore.scatter_(-1, idx1, 0)     # except the positive
            mask_id = mask.nonzero().squeeze()
        
        match_logits:torch.Tensor = match_logits - 1e4 * ignore.float()
        loss_feat = match_logits.logsumexp(dim=-1) - match_logits.gather(-1, idx1).squeeze(-1)
        loss_feat = loss_feat.index_select(0, mask_id).mean()
        if loss_feat.isnan(): loss_feat = 0.
        
        dist_keypts:torch.Tensor = torch.norm(src_xyz - tgt_corres, dim=-1)
        loss_corr = dist_keypts.index_select(0, mask_id).mean()
        if loss_corr.isnan(): loss_corr = 0.

        label = dist_keypts.lt(self.r_p)
        weight = torch.logical_not(label.logical_xor(dist_keypts.lt(self.r_n)))
        loss_ov = F.binary_cross_entropy(match_score, label.float(), weight.float())

        return loss_feat, loss_ov, loss_corr

    def forward(self, output_dict: Dict[str, torch.Tensor]):
        return self.cal_loss(
            output_dict['corres'][:, :3],
            output_dict['src_patch_corr_kpts'],
            output_dict['corres'][:, 3:],
            output_dict['match_logits'],
            output_dict['corr_confidence'],
            output_dict['gt_transform']
        )