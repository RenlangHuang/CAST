import os
import pickle
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

import torch
from torch.utils.data import Dataset


def read_3dmatch_bin_voxel(filename, npoints=None, voxel_size=None) -> np.ndarray:
    scan = np.array(o3d.io.read_point_cloud(filename).voxel_down_sample(voxel_size=voxel_size).points)
    scan = scan.astype('float32')
    if npoints is None:
        return scan
    
    if scan.shape[0] >= npoints:
        sample_idx = np.random.choice(scan.shape[0], npoints, replace=False)
        scan = scan[sample_idx, :]
    return scan


class IndoorDataset(Dataset):
    def __init__(self, root, seqs, npoints, voxel_size, data_list, augment=0.0):
        super(IndoorDataset, self).__init__()
        self.root = root
        self.seqs = seqs
        self.npoints = npoints
        self.voxel_size = voxel_size
        self.augment = augment
        self.data_list = data_list
        self.dataset = self.make_dataset()
    
    def make_dataset(self):
        last_row = np.zeros((1,4), dtype=np.float32)
        last_row[:,3] = 1.0
        dataset = []

        fn_pair_poses = os.path.join(self.data_list, self.seqs + '.pkl')
        fn_pair_poses = open(fn_pair_poses, 'rb')
        metadata = pickle.load(fn_pair_poses)
        fn_pair_poses.close()

        for i in range(len(metadata)):
            folder = os.path.join(self.root, 'train', metadata[i]['seq'], 'fragments')
            src_fn = os.path.join(folder, 'cloud_bin_%d.ply'%metadata[i]['ref_id'])
            dst_fn = os.path.join(folder, 'cloud_bin_%d.ply'%metadata[i]['src_id'])
            rela_pose = metadata[i]['transform'].astype(np.float32)
            rela_pose = np.concatenate([rela_pose, last_row], axis = 0)
            data_dict = {'points1': src_fn, 'points2': dst_fn, 'Tr': rela_pose}
            dataset.append(data_dict)

        return dataset
    
    def __getitem__(self, index):
        data_dict = self.dataset[index]
        src_points = read_3dmatch_bin_voxel(data_dict['points1'], self.npoints, self.voxel_size)
        dst_points = read_3dmatch_bin_voxel(data_dict['points2'], self.npoints, self.voxel_size)
        Tr = data_dict['Tr']
        
        if np.random.rand() < self.augment:
            aug_T = np.eye(4, dtype=np.float32)
            aug_T[:3,:3] = self.sample_random_rotation()
            dst_points = dst_points @ aug_T[:3,:3]
            Tr = Tr @ aug_T
        
        src_points = torch.from_numpy(src_points)
        dst_points = torch.from_numpy(dst_points)
        Tr = torch.from_numpy(Tr)
        return src_points, dst_points, Tr
    
    def sample_random_rotation(self, pitch_scale=np.pi/3., roll_scale=np.pi/4.):
        roll = np.random.uniform(-roll_scale, roll_scale)
        pitch = np.random.uniform(-pitch_scale, pitch_scale)
        r = R.from_euler('xyz', [roll, pitch, 0.], degrees=False)
        return r.as_matrix()
    
    def __len__(self):
        return len(self.dataset)


class IndoorTestDataset(Dataset):
    def __init__(self, root, seqs, npoints, voxel_size, data_list, non_consecutive=False):
        super(IndoorTestDataset, self).__init__()
        self.root = root
        self.seqs = seqs
        self.npoints = npoints
        self.data_list = data_list
        self.voxel_size = voxel_size
        self.non_consecutive = non_consecutive
        self.dataset = self.make_dataset()
    
    def make_dataset(self):
        dataset = []
        benchmark = os.path.join(self.data_list, 'benchmark', self.seqs)
        scenes = os.listdir(benchmark)
        scenes.sort()

        for seq in scenes:
            metaseq = os.path.join(benchmark, seq)
            pairs, trans = self.read_transformation_log(metaseq)
            pairs, covs = self.read_covariance_log(metaseq)
            for pair, rela_pose, cov in zip(pairs, trans, covs):
                folder = os.path.join(self.root, 'test', seq, 'fragments')
                src_fn = os.path.join(folder, 'cloud_bin_%d.ply'%pair[0])
                dst_fn = os.path.join(folder, 'cloud_bin_%d.ply'%pair[1])
                data_dict = {'points1': src_fn, 'points2': dst_fn, 'Tr': rela_pose, 'Cov': cov}
                dataset.append(data_dict)

        return dataset
    
    def __getitem__(self, index):
        data_dict = self.dataset[index]
        src_points = read_3dmatch_bin_voxel(data_dict['points1'], self.npoints, self.voxel_size)
        dst_points = read_3dmatch_bin_voxel(data_dict['points2'], self.npoints, self.voxel_size)
        Tr, Cov = data_dict['Tr'], data_dict['Cov']
        
        src_points = torch.from_numpy(src_points)
        dst_points = torch.from_numpy(dst_points)
        Tr = torch.from_numpy(Tr)
        Cov = torch.from_numpy(Cov)
        return src_points, dst_points, Tr, Cov
    
    def read_transformation_log(self, seq:str):
        with open(os.path.join(seq, 'gt.log')) as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        pairs, trans = list(), list()
        num_pairs = len(lines) // 5
        for i in range(num_pairs):
            line_id = i * 5
            if self.non_consecutive:
                item = lines[line_id].split()
                if int(item[1]) - int(item[0]) == 1: continue
            pairs.append(lines[line_id].split())
            pairs[-1] = [int(pairs[-1][0]), int(pairs[-1][1])]
            transform = list()
            for j in range(1, 5):
                transform.append(lines[line_id + j].split())
            trans.append(np.array(transform, dtype=np.float32))
        return np.array(pairs), np.array(trans).astype(np.float32)
    
    def read_covariance_log(self, seq:str):
        with open(os.path.join(seq, 'gt.info')) as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        pairs, cov = list(), list()
        num_pairs = len(lines) // 7
        for i in range(num_pairs):
            line_id = i * 7
            if self.non_consecutive:
                item = lines[line_id].split()
                if int(item[1]) - int(item[0]) == 1: continue
            pairs.append(lines[line_id].split())
            pairs[-1] = [int(pairs[-1][0]), int(pairs[-1][1])]
            covariance = list()
            for j in range(1, 7):
                covariance.append(lines[line_id + j].split())
            cov.append(np.array(covariance, dtype=np.float32))
        return np.array(pairs), np.array(cov).astype(np.float32)
    
    def __len__(self):
        return len(self.dataset)