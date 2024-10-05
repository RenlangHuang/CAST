import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import json
import pickle
import numpy as np
from typing import Dict
from munch import munchify

from data.kitti_data import KittiDataset
from data.eth_data import ETHDataset

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
    def __init__(self, cfg, uda_cfg):
        super().__init__(cfg)
        self.cfg, self.uda_cfg = cfg, uda_cfg
        if not os.path.exists(uda_cfg.snapshot_dir):
            os.makedirs(uda_cfg.snapshot_dir)
        
        train_dataset = KittiDataset(
            cfg.data.root, 'train', cfg.data.npoints, cfg.data.voxel_size, cfg.data_list, cfg.data.augment)
        val_dataset = KittiDataset(
            cfg.data.root, 'test', cfg.data.npoints, cfg.data.voxel_size, cfg.data_list, 0.0)
        self.tune_dataset = ETHDataset(
            uda_cfg.data.root, uda_cfg.data.npoints, uda_cfg.data.voxel_size, uda_cfg.data.augment)
        test_dataset = ETHDataset(
            uda_cfg.data.root, uda_cfg.data.npoints, uda_cfg.data.voxel_size, 0)
        
        self.train_loader = DataLoader(train_dataset, 1, num_workers=cfg.data.num_workers, shuffle=True, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, 1, num_workers=cfg.data.num_workers, shuffle=False, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, 1, num_workers=cfg.data.num_workers, shuffle=False, pin_memory=True)
        
        self.model = CAST(cfg.model).cuda()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=cfg.optim.step_size, gamma=cfg.optim.gamma)
        self.loss_func = OverallLoss(cfg.loss).cuda()
        self.evaluator = Evaluator(cfg.eval).cuda()
    
    def step(self, data_dict) -> Dict[str,torch.Tensor]:
        output_dict = self.model(*data_dict)
        loss_dict: Dict = self.loss_func(output_dict, self.epoch)
        with torch.no_grad():
            result_dict = self.evaluator(output_dict)
        loss_dict.update(result_dict)
        return loss_dict
    
    def train_epoch(self):
        self.optimizer.zero_grad()
        steps = len(self.train_loader)
        for iteration, data_dict in enumerate(self.train_loader):
            self.iteration += 1
            self.model.backbone.cfg = self.cfg.model
            data_dict = self.to_cuda(data_dict)
            result_dict = self.step(data_dict)
            result_dict['loss'].backward()

            if self.clip_grad is not None:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.clip_grad)
            
            self.optimizer.step()
            self.optimizer.zero_grad()

            print("Epoch %d [%d/%d]"%(self.epoch, iteration+1, steps), end=' ')
            for key, value in result_dict.items():
                print(key, "%.4f"%float(value), end='; ')
            print()
            
            self.model.backbone.cfg = self.uda_cfg.model
            index = self.iteration % len(self.tune_dataset)
            data_dict = self.tune_dataset.__getitem__(index)
            data_dict = [x.cuda().unsqueeze(0) for x in data_dict]
            result_dict = self.step(data_dict)
            result_dict['loss'].backward()

            if self.clip_grad is not None:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.clip_grad)
            
            self.optimizer.step()
            self.optimizer.zero_grad()

            result_dict = self.release_cuda(result_dict)
            self.summary_board.update_from_dict(result_dict)
            #torch.cuda.empty_cache()

            print("      %d [%d/%d]"%(self.epoch, iteration+1, steps), end=' ')
            for key, value in result_dict.items():
                print(key, "%.4f"%float(value), end='; ')
            print()

            if (iteration + 1) % self.log_steps == 0:
                logs = dict()
                for k,v in self.summary_board.meter_dict.items():
                    logs[k] = np.array(v._records)
                print("Logging into ", self.log_file)
                flog = open(self.log_file, 'wb')
                flog.write(pickle.dumps(logs))
                flog.close()
            
            if self.save_steps > 0 and (iteration + 1) % self.save_steps == 0:
                self.save_snapshot("-epoch-%02d-%d"%(self.epoch, iteration + 1))
                if self.scheduler is not None: self.scheduler.step()
        
        if self.scheduler is not None:
            self.scheduler.step()


if __name__ == "__main__":
    with open('./config/kitti.json', 'r') as cfg:
        args = json.load(cfg)
        args = munchify(args)
    with open('./config/eth.json', 'r') as cfg:
        uda_args = json.load(cfg)
        uda_args = munchify(uda_args)
    
    engine = Trainer(args, uda_args)
    engine.load_snapshot('cast-epoch-39')
    engine.snapshot_dir = uda_args.snapshot_dir
    engine.max_epoch = engine.epoch + uda_args.optim.max_epoch
    for i in range(uda_args.optim.max_epoch):
        engine.train_epoch()
        engine.save_snapshot("-epoch-%02d"%(i+1))