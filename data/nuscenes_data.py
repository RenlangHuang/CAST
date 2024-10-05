import os
import torch
import numpy as np
import MinkowskiEngine as ME
from torch.utils.data import Dataset
from models.utils import generate_rand_rotm

def read_nuscenes_bin_voxel(filename, npoints=None, voxel_size=None) -> np.ndarray:
    scan = np.fromfile(filename, dtype=np.float32, count=-1).reshape([-1,5])
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

    scan = scan[np.logical_and(dist > 3., scan[:, 2] > -6.)]
    return scan


class NuscenesDataset(Dataset):
    def __init__(self, root, seqs, npoints, voxel_size, data_list, augment=0.0):
        super(NuscenesDataset, self).__init__()

        self.root = root
        self.seqs = seqs
        self.npoints = npoints
        self.voxel_size = voxel_size
        self.data_list = data_list
        self.augment = augment
        self.dataset = self.make_dataset()
    
    def make_dataset(self):
        last_row = np.zeros((1,4), dtype=np.float32)
        last_row[:,3] = 1.0
        dataset = []

        if True:#for seq in self.seqs:
            data_root = self.root
            '''if seq == 'test':
                data_root = os.path.join(self.root, 'v1.0-test')
            else:
                data_root = os.path.join(self.root, 'v1.0-trainval')'''
            fn_pair_poses = os.path.join(self.data_list, self.seqs + '.txt')
            with open(fn_pair_poses, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    data_dict = {}
                    line = line.strip(' \n').split(' ')
                    src_fn = os.path.join(data_root, line[0])
                    dst_fn = os.path.join(data_root, line[1])
                    values = []
                    for i in range(2, len(line)):
                        values.append(float(line[i]))
                    values = np.array(values).astype(np.float32)
                    rela_pose = values.reshape(3,4)
                    rela_pose = np.concatenate([rela_pose, last_row], axis = 0)
                    data_dict['points1'] = src_fn
                    data_dict['points2'] = dst_fn
                    data_dict['Tr'] = rela_pose
                    dataset.append(data_dict)
        
        return dataset
    
    def __getitem__(self, index):
        data_dict = self.dataset[index]
        src_points = read_nuscenes_bin_voxel(data_dict['points1'], self.npoints, self.voxel_size)
        dst_points = read_nuscenes_bin_voxel(data_dict['points2'], self.npoints, self.voxel_size)
        Tr = np.linalg.inv(data_dict['Tr'])

        if np.random.rand() < self.augment:
            aug_T = np.eye(4, dtype=np.float32)
            aug_T[:3,:3] = generate_rand_rotm(1.0, 1.0)
            dst_points = dst_points @ aug_T[:3,:3]
            Tr = Tr @ aug_T # np.linalg.inv(aug_T)
        
        src_points = torch.from_numpy(src_points)
        dst_points = torch.from_numpy(dst_points)
        Tr = torch.from_numpy(Tr)
        return src_points, dst_points, Tr
    
    def __len__(self):
        return len(self.dataset)