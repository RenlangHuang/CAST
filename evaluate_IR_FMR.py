import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'

import json
import torch
import argparse
from munch import munchify
from torch.utils.data import DataLoader

from data.indoor_data import IndoorTestDataset
from engine.trainer import EpochBasedTrainer

from models.models.cast import CAST
from engine.evaluator import Evaluator


parser = argparse.ArgumentParser()
parser.add_argument("--benchmark", default='3DMatch', choices=['3DMatch', '3DLoMatch'])
parser.add_argument("--config", default='./config/3dmatch.json', type=str)
parser.add_argument("--load_pretrained", default='cast-epoch-05', type=str)
parser.add_argument("--ransac", default=False, action="store_true")

_args = parser.parse_args()


class Tester(EpochBasedTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        val_dataset = IndoorTestDataset(cfg.data.root, _args.benchmark, cfg.data.npoints, cfg.data.voxel_size, cfg.data_list)
        self.val_loader = DataLoader(val_dataset, 1, num_workers=cfg.data.num_workers, shuffle=False, pin_memory=True)
        
        self.model = CAST(cfg.model).cuda()
        self.evaluator = Evaluator(cfg.eval).cuda()
    
    def step(self, data_dict):
        output_dict = self.model(*data_dict[:3])
        trans = output_dict['gt_transform']
        PIR = self.evaluator.evaluate_coarse_inlier_ratio(output_dict)
        results = {'PIR': PIR, 'PMR': PIR.gt(0.2).float()}
        
        indices = torch.argsort(output_dict['corr_confidence'], descending=True)
        corr_confidence = output_dict['corr_confidence'][indices]
        corres = output_dict['corres'][indices]
        indices = indices[corr_confidence.gt(0.1 * corr_confidence[0])]
        corres_ = output_dict['corres'][indices]
        for num in [250, 500, 1000, 2500, 5000]:
            results['IR@%d'%num] = self.evaluator.evaluate_inlier_ratio(corres_[:num], trans)
            results['FMR@%d'%num] = self.evaluator.evaluate_inlier_ratio(corres[:num], trans).gt(0.05).float()
        return results


if __name__ == "__main__":
    with open(_args.config, 'r') as cfg:
        args = json.load(cfg)
        args = munchify(args)
    args.model.ransac = False

    tester = Tester(args)
    tester.set_eval_mode()
    tester.load_snapshot(_args.load_pretrained)
    # e.g. tester.load_snapshot("cast-epoch-05")
    tester.validate_epoch()