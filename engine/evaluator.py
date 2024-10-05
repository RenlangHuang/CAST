import torch
import numpy as np
import torch.nn as nn
from models.utils import apply_transform
from scipy.spatial.transform import Rotation as R


class Evaluator(nn.Module):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        self.hit_threshold = cfg.hit_threshold
        self.acceptance_overlap = cfg.acceptance_overlap
        self.inlier_ratio_threshold = 0.05
        if 'rre_threshold' in cfg.keys():
            self.rre_threshold = cfg.rre_threshold
            self.rte_threshold = cfg.rte_threshold
            self.scene = 'outdoor'
        else:
            self.inlier_distance_threshold = 0.1
            self.rmse_threshold = cfg.rmse_threshold
            self.scene = 'indoor'
    
    @torch.no_grad()
    def keypoint_repeatability(self, keypoints1: torch.Tensor, keypoints2: torch.Tensor, transform: torch.Tensor):
        keypoints2 = apply_transform(keypoints2, transform)  # (M, 3)
        dist = torch.norm(keypoints1.unsqueeze(1) - keypoints2.unsqueeze(0), dim=-1) # (M, N)
        forward_KR =  torch.min(dist, dim=-1)[0].lt(self.hit_threshold).float().mean()
        backward_KR = torch.min(dist, dim=-2)[0].lt(self.hit_threshold).float().mean()
        return (forward_KR + backward_KR) / 2.
    
    @torch.no_grad()
    def evaluate_inlier_ratio(self, corr_points: torch.Tensor, transform: torch.Tensor):
        src_corr_points = apply_transform(corr_points[..., 3:], transform)
        dist = torch.norm(src_corr_points - corr_points[..., :3], dim=-1)
        return torch.lt(dist, self.inlier_distance_threshold).float().mean()
    
    @torch.no_grad()
    def compute_rmse(self, transform, covariance, estimated_transform):
        relative_transform = torch.matmul(torch.linalg.inv(transform), estimated_transform)
        q = R.from_matrix(relative_transform[:3, :3].cpu().numpy()).as_quat()
        q = torch.from_numpy(q[:3]).float().to(transform.device)
        er = torch.cat([relative_transform[:3, 3], q], dim=-1)
        er = er.view(1, 6) @ covariance @ er.view(6, 1) / covariance[0, 0]
        return torch.sqrt(er)

    @torch.no_grad()
    def transform_error(self, gt_transforms: torch.Tensor, transforms: torch.Tensor):
        rre = 0.5 * ((transforms[:3, :3].T @ gt_transforms[:3, :3]).trace() - 1.0)
        rre = 180.0 * torch.arccos(rre.clamp(-1., 1.)) / np.pi
        rte = torch.norm(gt_transforms[:3, 3] - transforms[:3, 3], dim=-1)
        return rte, rre

    @torch.no_grad()
    def evaluate_coarse_inlier_ratio(self, output_dict):
        ref_length_c = output_dict['ref_feats_c'].shape[0]
        src_length_c = output_dict['src_feats_c'].shape[0]
        masks = torch.gt(output_dict['gt_patch_corr_overlaps'], self.acceptance_overlap)
        gt_node_corr_indices = output_dict['gt_patch_corr_indices'][masks]
        gt_node_corr_map = torch.zeros([ref_length_c, src_length_c], device=masks.device)
        gt_node_corr_map[gt_node_corr_indices[:, 0], gt_node_corr_indices[:, 1]] = 1.0
        return gt_node_corr_map[output_dict['ref_patch_corr_indices'], output_dict['src_patch_corr_indices']].mean()

    @torch.no_grad()
    def evaluate_registration(self, output_dict):
        if self.scene == 'outdoor':
            rte, rre = self.transform_error(output_dict['gt_transform'], output_dict['transform'])
            recall = torch.logical_and(torch.lt(rre, self.rre_threshold), torch.lt(rte, self.rte_threshold)).float()
            return rre, rte, recall
        else:
            rmse = self.compute_rmse(output_dict['gt_transform'], output_dict['covariance'], output_dict['transform'])
            return rmse, rmse < self.rmse_threshold

    def forward(self, output_dict):
        PIR = self.evaluate_coarse_inlier_ratio(output_dict)
        te, re = self.transform_error(output_dict['gt_transform'], output_dict['transform'])
        rte, rre = self.transform_error(output_dict['gt_transform'], output_dict['refined_transform'])
        #KR = self.keypoint_repeatability(output_dict['ref_kpts'], output_dict['src_kpts'], output_dict['gt_transform'])
        results = {'PIR': PIR} #{'KR': KR, 'PIR': PIR}
        
        if not self.training:
            results['PMR'] = PIR.gt(0.2).float()

        if self.scene == 'indoor':
            indices = output_dict['corr_confidence'] > 0.1 * torch.max(output_dict['corr_confidence'])
            results['IR'] = self.evaluate_inlier_ratio(output_dict['corres'][indices], output_dict['gt_transform'])
            FMR = results['IR'].gt(self.inlier_distance_threshold).float()
            results['RE'] = re; results['TE'] = te
            results['RRE'] = rre; results['RTE'] = rte
            if not self.training:
                results['FMR'] = FMR
            if 'covariance' in output_dict.keys():
                covariance = output_dict['covariance']
                gt_transform = output_dict['gt_transform']
                #pred_transform = output_dict['transform']
                pred_transform = output_dict['refined_transform']
                results['rmse'] = self.compute_rmse(gt_transform, covariance, pred_transform)
                results['RR'] = results['rmse'].lt(self.rmse_threshold).float()
                if results['RR'] < 0.5:
                    results.pop('RTE'); results.pop('RRE')
                    results.pop('TE'); results.pop('RE')
        
        else:
            registration_recall = torch.lt(rre, self.rre_threshold) & torch.lt(rte, self.rte_threshold)
            if self.training or registration_recall.item():
                results['RE'] = re; results['TE'] = te
                results['RRE'] = rre; results['RTE'] = rte
            if not self.training:
                results['RR'] = registration_recall.float()
        
        return results
