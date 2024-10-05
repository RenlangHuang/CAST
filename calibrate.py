import torch
import random
import numpy as np
from munch import Munch
from data.indoor_data import IndoorDataset
from data.kitti_data import KittiDataset
from data.nuscenes_data import NuscenesDataset
from models.utils import grid_subsample_gpu, radius_search_gpu


def collate_fn(points, lengths, num_stages, voxel_size, radius, max_neighbor_limits):
    neighbors_list = list()

    for i in range(num_stages):
        neighbors_list.append(radius_search_gpu(points, points, lengths, lengths, radius, max_neighbor_limits[i]))
        radius, voxel_size = radius * 2., voxel_size * 2.
        if i == num_stages - 1: break
        points, lengths = grid_subsample_gpu(points, lengths, voxel_size)
    
    return neighbors_list


def calibrate_neighbors_stack_mode(
    dataset, num_stages, voxel_size, search_radius, keep_ratio=0.8, sample_threshold=2000
):
    # Compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (search_radius / voxel_size + 1) ** 3))
    neighbor_hists = np.zeros((num_stages, hist_n), dtype=np.int32)
    max_neighbor_limits = [hist_n] * num_stages

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):
        data_dict = dataset[i]
        points = torch.cat([data_dict[0], data_dict[1]], dim=0)
        lengths = torch.LongTensor([data_dict[0].shape[0], data_dict[1].shape[0]])
        data_dict = collate_fn(points, lengths, num_stages, voxel_size, search_radius, max_neighbor_limits)
        counts = [np.sum(neighbors.numpy() < neighbors.shape[0], axis=1) for neighbors in data_dict]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighbor_hists += np.vstack(hists)
        if np.min(np.sum(neighbor_hists, axis=1)) > sample_threshold: break

    cum_sum = np.cumsum(neighbor_hists.T, axis=0)
    return np.sum(cum_sum < (keep_ratio * cum_sum[hist_n - 1, :]), axis=0)


info = Munch()
info.indoor = Munch()
info.indoor.root = '/home/jacko/Downloads/3dmatch/'
info.indoor.data_list = './data/3dmatch_list/'
info.kitti = Munch()
info.kitti.root = '/media/jacko/SSD/KITTI/velodyne/sequences/'
info.kitti.data_list = './data/kitti_list/'
info.nuscenes = Munch()
info.nuscenes.root = '/media/jacko/SSD/nuscenes/'
info.nuscenes.data_list = './data/nuscenes_list/'

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

dataset = KittiDataset(info.kitti.root, 'train', 30000, 0.3, info.kitti.data_list, 0.5)
print('Create a data loader for KITTI with %d samples.'%len(dataset))
neighbor_limits = calibrate_neighbors_stack_mode(dataset, 5, 0.3, 0.3 * 3.0)
print('Calibrate neighbors for KITTI: ', neighbor_limits)

dataset = NuscenesDataset(info.nuscenes.root, 'train', 30000, 0.3, info.nuscenes.data_list, 0.5)
print('Create a data loader for NuScenes with %d samples.'%len(dataset))
neighbor_limits = calibrate_neighbors_stack_mode(dataset, 5, 0.3, 0.3 * 3.25)
print('Calibrate neighbors for NuScenes: ', neighbor_limits)

dataset = IndoorDataset(info.indoor.root, 'train', None, 0.03, info.indoor.data_list, 0.5)
print('Create a data loader for 3DMatch with %d samples.'%len(dataset))
neighbor_limits = calibrate_neighbors_stack_mode(dataset, 4, 0.03, 0.025 * 2.5)
print('Calibrate neighbors for 3DMatch: ', neighbor_limits)

'''
Outputs:
    Create a data loader for KITTI with 1358 samples.
    Calibrate neighbors for KITTI:  [35 35 36 37 41]
    Create a data loader for NuScenes with 26696 samples.
    Calibrate neighbors for NuScenes:  [19 26 36 45 46]
    Create a data loader for 3DMatch with 20642 samples.
    Calibrate neighbors for 3DMatch:  [26 21 22 25]
'''