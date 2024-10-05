import numpy as np
import torch
from torch.utils.data import Dataset

import os
import numpy as np
import MinkowskiEngine as ME

from models.utils import generate_rand_rotm


def read_kitti_bin_voxel(filename, npoints=None, voxel_size=None) -> np.ndarray:
    scan = np.fromfile(filename, dtype=np.float32, count=-1).reshape([-1,4])
    scan = scan[:,:3]

    if voxel_size is not None:
        _, sel = ME.utils.sparse_quantize(scan / voxel_size, return_index=True)
        scan = scan[sel]
    if npoints is None:
        return scan.astype('float32')
    
    dist = np.linalg.norm(scan, ord=2, axis=1)
    N = scan.shape[0]
    if N >= npoints:
        sample_idx = np.argsort(dist)[:npoints]
        scan = scan[sample_idx, :].astype('float32')
        dist = dist[sample_idx]
    scan = scan[np.logical_and(dist > 3., scan[:, 2] > -3.5)]
    return scan

class KittiDataset(Dataset):
    def __init__(self, root, seqs, npoints, voxel_size, data_list, augment=0.0):
        super(KittiDataset, self).__init__()
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

        fn_pair_poses = os.path.join(self.data_list, self.seqs + '.txt')
        metadata = np.genfromtxt(fn_pair_poses).reshape([-1, 15])
        for i in range(metadata.shape[0]):
            folder = os.path.join(self.root, '%02d'%metadata[i][0], 'velodyne')
            src_fn = os.path.join(folder, '%06d.bin'%metadata[i][1])
            dst_fn = os.path.join(folder, '%06d.bin'%metadata[i][2])
            rela_pose = metadata[i][3:].reshape(3,4).astype(np.float32)
            rela_pose = np.concatenate([rela_pose, last_row], axis = 0)
            data_dict = {'points1': src_fn, 'points2': dst_fn, 'Tr': rela_pose}
            dataset.append(data_dict)

        return dataset
    
    def __getitem__(self, index):
        data_dict = self.dataset[index]
        src_points = read_kitti_bin_voxel(data_dict['points1'], self.npoints, self.voxel_size)
        dst_points = read_kitti_bin_voxel(data_dict['points2'], self.npoints, self.voxel_size)
        Tr = data_dict['Tr']
        
        if np.random.rand() < self.augment:
            aug_T = np.eye(4, dtype=np.float32)
            aug_T[:3,:3] = generate_rand_rotm(1.0, 1.0)
            dst_points = dst_points @ aug_T[:3,:3]
            Tr = Tr @ aug_T
        
        src_points = torch.from_numpy(src_points)
        dst_points = torch.from_numpy(dst_points)
        Tr = torch.from_numpy(Tr)
        return src_points, dst_points, Tr
    
    def __len__(self):
        return len(self.dataset)