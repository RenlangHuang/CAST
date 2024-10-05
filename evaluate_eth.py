import json
import argparse
import numpy as np
from munch import munchify

import torch
from torch.utils.data import DataLoader

from data.eth_data import ETHDataset
from engine.trainer import EpochBasedTrainer

from models.models.cast_eth import CAST
from engine.evaluator import Evaluator


parser = argparse.ArgumentParser()
parser.add_argument("--load_pretrained", default='kitti/cast-epoch-39.pth.tar', type=str)
parser.add_argument("--filter", default='DSC', choices=['DSC', 'SM'])

_args = parser.parse_args()


class Tester(EpochBasedTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        val_dataset = ETHDataset(cfg.data.root, cfg.data.npoints, cfg.data.voxel_size)
        self.val_loader = DataLoader(val_dataset, 1, num_workers=cfg.data.num_workers, shuffle=False, pin_memory=True)
        self.val_dataset = val_dataset
        
        cfg.model.filter = _args.filter
        self.model = CAST(cfg.model).cuda()
        self.evaluator = Evaluator(cfg.eval).cuda()
    
    def step(self, data_dict):
        output_dict = self.model(*data_dict[:3])
        gt_trans = output_dict['gt_transform']
        pred_trans = output_dict['refined_transform']
        rte, rre = self.evaluator.transform_error(gt_trans, pred_trans)
        return {"RTE": rte, "RRE": rre}


if __name__ == "__main__":
    with open('./config/eth.json', 'r') as cfg:
        args = json.load(cfg)
        args = munchify(args)
    
    tester = Tester(args)
    tester.set_eval_mode()
    state_dict = torch.load('./ckpt/' + _args.load_pretrained)
    tester.model.load_state_dict(state_dict['model'], strict=False)
    result_list = tester.validate_epoch()

    RTE = np.array(result_list.meter_dict['RTE'].records)
    RRE = np.array(result_list.meter_dict['RRE'].records)

    RR = np.logical_and(RTE < 0.3, RRE < 2.0).astype(dtype=RTE.dtype)
    print("Threshold TE@0.3m,RE@2°", end=", ")
    print('RR: %.4f, RRE: %.4f, RTE: %.4f'%(RR.mean(), (RRE * RR).sum() / RR.sum(), (RTE * RR).sum() / RR.sum()))
    RR = np.logical_and(RTE < 0.3, RRE < 5.0).astype(dtype=RTE.dtype)
    print("Threshold TE@0.3m,RE@5°", end=", ")
    print('RR: %.4f, RRE: %.4f, RTE: %.4f'%(RR.mean(), (RRE * RR).sum() / RR.sum(), (RTE * RR).sum() / RR.sum()))
    RR = np.logical_and(RTE < 2.0, RRE < 5.0).astype(dtype=RTE.dtype)
    print("Threshold TE@2.0m,RE@5°", end=", ")
    print('RR: %.4f, RRE: %.4f, RTE: %.4f'%(RR.mean(), (RRE * RR).sum() / RR.sum(), (RTE * RR).sum() / RR.sum()))


# RANSAC
#Threshold TE@0.3m,RE@2°, RR: 0.9158, RRE: 0.5176, RTE: 0.0640
#Threshold TE@0.3m,RE@5°, RR: 0.9705, RRE: 0.6422, RTE: 0.0699
#Threshold TE@2.0m,RE@5°, RR: 0.9846, RRE: 0.6652, RTE: 0.0743
# RANSAC + DSC
#Threshold TE@0.3m,RE@2°, RR: 0.9285, RRE: 0.5409, RTE: 0.0626
#Threshold TE@0.3m,RE@5°, RR: 0.9776, RRE: 0.6499, RTE: 0.0685
#Threshold TE@2.0m,RE@5°, RR: 0.9832, RRE: 0.6593, RTE: 0.0704
# RANSAC + SM
#Threshold TE@0.3m,RE@2°, RR: 0.9467, RRE: 0.5428, RTE: 0.0630
#Threshold TE@0.3m,RE@5°, RR: 0.9804, RRE: 0.6095, RTE: 0.0666
#Threshold TE@2.0m,RE@5°, RR: 0.9874, RRE: 0.6279, RTE: 0.0690