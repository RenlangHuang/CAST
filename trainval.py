import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'


import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import json
from typing import Dict
from munch import munchify

from data.kitti_data import KittiDataset
from data.nuscenes_data import NuscenesDataset
from data.indoor_data import IndoorDataset

from models.models.cast import CAST
from engine.evaluator import Evaluator
from engine.trainer import EpochBasedTrainer
from engine.losses import SpotMatchingLoss, CoarseMatchingLoss, KeypointMatchingLoss, ProbChamferLoss


class OverallLoss(torch.nn.Module):
    def __init__(self, cfg):
        super(OverallLoss, self).__init__()
        self.weight_det_loss = cfg.weight_det_loss
        self.weight_spot_loss = cfg.weight_spot_loss
        self.weight_feat_loss = cfg.weight_feat_loss
        self.weight_desc_loss = cfg.weight_desc_loss
        self.weight_overlap_loss = cfg.weight_overlap_loss
        self.weight_corr_loss = cfg.weight_corr_loss
        self.weight_trans_loss = cfg.weight_trans_loss
        self.weight_rot_loss = cfg.weight_rot_loss
        self.pretrain_feat_epochs = cfg.pretrain_feat_epochs
        
        self.prob_chamfer_loss = ProbChamferLoss()
        self.spot_matching_loss = SpotMatchingLoss(cfg)
        self.coarse_matching_loss = CoarseMatchingLoss(cfg)
        self.kpt_matching_loss = KeypointMatchingLoss(cfg.r_p, cfg.r_n)
        self.register_buffer('I3x3', torch.eye(3))
    
    def forward(self, output_dict: Dict[str, torch.Tensor], epoch) -> Dict[str, torch.Tensor]:
        l_det = self.prob_chamfer_loss(output_dict)
        l_spot,l_feat = self.spot_matching_loss(output_dict)
        #l_feat = self.coarse_matching_loss(output_dict)
        loss = l_feat * self.weight_feat_loss + l_det * self.weight_det_loss + l_spot * self.weight_spot_loss
        
        loss_dict = {'l_det':l_det, 'l_spot':l_spot, 'l_feat':l_feat}

        l_desc, l_ov, l_corr = self.kpt_matching_loss(output_dict)
        l_trans = torch.norm(output_dict['transform'][:3, 3] - output_dict['gt_transform'][:3, 3])
        l_rot = torch.norm(output_dict['transform'][:3, :3].T @ output_dict['gt_transform'][:3, :3] - self.I3x3)
        loss = loss + l_desc * self.weight_desc_loss + l_ov * self.weight_overlap_loss + l_corr * self.weight_corr_loss
        
        loss_dict.update({'l_corr':l_corr, 'l_desc':l_desc, 'l_ov':l_ov})

        l_trans2 = torch.norm(output_dict['refined_transform'][:3, 3] - output_dict['gt_transform'][:3, 3])
        l_rot2 = torch.norm(output_dict['refined_transform'][:3, :3].T @ output_dict['gt_transform'][:3, :3] - self.I3x3)

        if epoch > self.pretrain_feat_epochs:
            loss = loss + l_trans.clamp_max(2.) * self.weight_trans_loss + l_rot.clamp_max(1.) * self.weight_rot_loss
            loss = loss + l_trans2.clamp_max(2.) * self.weight_trans_loss + l_rot2.clamp_max(1.) * self.weight_rot_loss
        
        ret_dict = {'loss':loss, 'l_rot':l_rot}
        ret_dict.update(loss_dict)
        return ret_dict


class Trainer(EpochBasedTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.dataset == 'kitti':
            train_dataset = KittiDataset(
                cfg.data.root, 'train', cfg.data.npoints, cfg.data.voxel_size, cfg.data_list, cfg.data.augment)
            val_dataset = KittiDataset(
                cfg.data.root, 'test', cfg.data.npoints, cfg.data.voxel_size, cfg.data_list, 0.0)
        elif cfg.dataset == 'nuscenes':
            train_dataset = NuscenesDataset(
                cfg.data.root, 'train', cfg.data.npoints, cfg.data.voxel_size, cfg.data_list, cfg.data.augment)
            val_dataset = NuscenesDataset(
                cfg.data.root, 'test', cfg.data.npoints, cfg.data.voxel_size, cfg.data_list, 0.0)
        elif cfg.dataset == '3dmatch':
            train_dataset = IndoorDataset(
                cfg.data.root, 'train', cfg.data.npoints, cfg.data.voxel_size, cfg.data_list, cfg.data.augment)
            val_dataset = IndoorDataset(
                cfg.data.root, 'val', cfg.data.npoints, cfg.data.voxel_size, cfg.data_list, 0.0)
        else:
            raise('Not implemented')
        
        self.train_loader = DataLoader(train_dataset, 1, num_workers=cfg.data.num_workers, shuffle=True, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, 1, num_workers=cfg.data.num_workers, shuffle=False, pin_memory=True)
        
        self.model = CAST(cfg.model).cuda()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=cfg.optim.step_size, gamma=cfg.optim.gamma)
        self.loss_func = OverallLoss(cfg.loss).cuda()
        self.evaluator = Evaluator(cfg.eval).cuda()

        self.pretrain_feat_epochs = cfg.loss.pretrain_feat_epochs
    
    def step(self, data_dict) -> Dict[str,torch.Tensor]:
        output_dict = self.model(*data_dict, self.epoch <= self.pretrain_feat_epochs)
        loss_dict: Dict = self.loss_func(output_dict, self.epoch)
        with torch.no_grad():
            result_dict = self.evaluator(output_dict)
        loss_dict.update(result_dict)
        return loss_dict



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, default="train", choices=["train", "test"])
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--resume_epoch", default=0, type=int)
    parser.add_argument("--resume_log", default=None, type=str)
    parser.add_argument("--load_pretrained", default=None, type=str)

    _args = parser.parse_args()

    with open(_args.config, 'r') as cfg:
        args = json.load(cfg)
        args = munchify(args)
    
    if _args.mode == "train":
        Trainer(args).fit(_args.resume_epoch, _args.resume_log)
    elif _args.mode == "test":
        tester = Trainer(args)
        tester.load_snapshot(_args.load_pretrained)
        # e.g. tester.load_snapshot("cast-epoch-39")
        tester.validate_epoch()
    else: assert "Unspecified mode."