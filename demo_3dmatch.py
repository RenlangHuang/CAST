import os
import json
import torch
import argparse
import numpy as np
import open3d as o3d
from munch import munchify
from engine.trainer import EpochBasedTrainer
from data.indoor_data import IndoorDataset, IndoorTestDataset
from models.models.cast import CAST


parser = argparse.ArgumentParser()
parser.add_argument("--split", default='train', choices=['train', 'val', 'test'])
parser.add_argument("--benchmark", default='3DMatch', choices=['3DMatch', '3DLoMatch'])
parser.add_argument("--load_pretrained", default='cast-epoch-05', type=str)
parser.add_argument("--id", default=0, type=int)

_args = parser.parse_args()


class Engine(EpochBasedTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.train_dataset = IndoorDataset(cfg.data.root, 'train', cfg.data.npoints, cfg.data.voxel_size, cfg.data_list, 0.0)
        self.val_dataset = IndoorDataset(cfg.data.root, 'val', cfg.data.npoints, cfg.data.voxel_size, cfg.data_list, 0.0)
        self.test_dataset = IndoorTestDataset(cfg.data.root, _args.benchmark, cfg.data.npoints, cfg.data.voxel_size, cfg.data_list)
        
        self.model = CAST(cfg.model).cuda()


if __name__ == "__main__":
    with open('./config/3dmatch.json', 'r') as cfg:
        args = json.load(cfg)
        args = munchify(args)
    
    tester = Engine(args)
    tester.set_eval_mode()
    tester.load_snapshot(_args.load_pretrained)

    if _args.split == 'train':
        data = tester.train_dataset[_args.id]
        data_dict = tester.train_dataset.dataset[_args.id]
    elif _args.split == 'val':
        data = tester.val_dataset[_args.id]
        data_dict = tester.val_dataset.dataset[_args.id]
    else:
        data = tester.test_dataset[_args.id]
        data_dict = tester.test_dataset.dataset[_args.id]
    
    gt_trans = data[2].numpy()
    ref_cloud = o3d.io.read_point_cloud(data_dict['points1'])
    src_cloud = o3d.io.read_point_cloud(data_dict['points2'])

    custom_yellow = np.asarray([[221., 184., 34.]]) / 255.0
    custom_blue = np.asarray([[9., 151., 247.]]) / 255.0
    ref_cloud.paint_uniform_color(custom_blue.T)
    src_cloud.paint_uniform_color(custom_yellow.T)
    
    data = [v.cuda().unsqueeze(0) for v in data]
    with torch.no_grad():
        output_dict = tester.model(*data)
        trans = output_dict['refined_transform'].cpu().numpy()

    ref_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    src_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    src_cloud.transform(trans)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    view_option: o3d.visualization.ViewControl = vis.get_view_control()
    render_option: o3d.visualization.RenderOption = vis.get_render_option()	
    render_option.background_color = np.array([0, 0, 0])
    render_option.background_color = np.array([1, 1, 1])
    render_option.point_size = 3.0
    vis.add_geometry(ref_cloud)
    vis.add_geometry(src_cloud)
    view_option.set_front([0., -0.3, -1.])
    view_option.set_up([0., 0., 1.])
    vis.run()