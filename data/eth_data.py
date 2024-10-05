import os
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset
from models.utils import generate_rand_rotm


def read_eth_bin_voxel(filename, npoints=None, voxel_size=None) -> np.ndarray:
    scan = np.array(o3d.io.read_point_cloud(filename).voxel_down_sample(voxel_size=voxel_size).points)
    scan = scan.astype('float32')
    if npoints is None:
        return scan
    
    if scan.shape[0] >= npoints:
        sample_idx = np.random.choice(scan.shape[0], npoints, replace=False)
        scan = scan[sample_idx, :]
    return scan


class ETHDataset(Dataset):
    def __init__(self, root, npoints, voxel_size, augment=0.0):
        super(ETHDataset, self).__init__()
        self.root = root
        self.npoints = npoints
        self.voxel_size = voxel_size
        self.augment = augment
        self.dataset = self.make_dataset()
    
    def make_dataset(self):
        scenes = [
            'gazebo_summer',
            'gazebo_winter',
            'wood_autmn',
            'wood_summer',
        ]
        dataset = []

        for seq in scenes:
            folder = os.path.join(self.root, seq)
            pairs, trans = self.read_transformation_log(folder)
            for pair, rela_pose in zip(pairs, trans):
                src_fn = os.path.join(folder, 'Hokuyo_%d.ply'%pair[0])
                dst_fn = os.path.join(folder, 'Hokuyo_%d.ply'%pair[1])
                data_dict = {'points1': src_fn, 'points2': dst_fn, 'Tr': rela_pose}
                dataset.append(data_dict)

        return dataset
    
    def __getitem__(self, index):
        data_dict = self.dataset[index]
        src_points = read_eth_bin_voxel(data_dict['points1'], self.npoints, self.voxel_size)
        dst_points = read_eth_bin_voxel(data_dict['points2'], self.npoints, self.voxel_size)
        Tr = data_dict['Tr']

        if self.augment > 0:
            cross = np.random.uniform(low=dst_points.min(0), high=dst_points.max(0))
            if cross[0] <  0.1 and cross[0] > 0: cross[0] =  0.1
            if cross[0] > -0.1 and cross[0] < 0: cross[0] = -0.1
            if cross[1] <  0.1 and cross[1] > 0: cross[1] =  0.1
            if cross[1] > -0.1 and cross[1] < 0: cross[1] = -0.1
            crop = dst_points[:, 0] / cross[0] + dst_points[:, 1] / cross[1]
            src_points = dst_points[crop < 1]
            Tr = np.eye(4, dtype=np.float32)
        
        if np.random.rand() < self.augment:
            print(src_points.shape)
            aug_T = np.eye(4, dtype=np.float32)
            aug_T[:2, 3] = np.random.random([2]) * 2. - 1.
            aug_T[:3,:3] = generate_rand_rotm(0., 0.)
            src_points = src_points @ aug_T[:3,:3].T + aug_T[:3,3:].T # dst_points
            Tr = aug_T
        
        src_points = torch.from_numpy(src_points)
        dst_points = torch.from_numpy(dst_points)
        Tr = torch.from_numpy(Tr)
        return src_points, dst_points, Tr
    
    def __len__(self):
        return len(self.dataset)
    
    def read_transformation_log(self, seq:str):
        with open(os.path.join(seq, 'gt.log')) as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        pairs, trans = list(), list()
        num_pairs = len(lines) // 5
        for i in range(num_pairs):
            line_id = i * 5
            pairs.append(lines[line_id].split())
            pairs[-1] = [int(pairs[-1][0]), int(pairs[-1][1])]
            transform = list()
            for j in range(1, 5):
                transform.append(lines[line_id + j].split())
            trans.append(np.array(transform, dtype=np.float32))
        return np.array(pairs), np.array(trans).astype(np.float32)