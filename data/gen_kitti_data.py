"""
This code is modified from D3Feat, which can generate
point cloud pairs for kitti and refine the ground-truth transformations.
If you want to regenerate these results, please modify the directories below.
"""
import os
import glob
import numpy as np
import open3d as o3d


class KITTIDataset:
    DATA_FILES = {
        'train': ['00', '01', '02', '03', '04', '05'],
        'val': ['06', '07'],
        'test': ['08', '09', '10']
    }
    def __init__(self):
        self.root = '/media/jacko/SSD/KITTI/'
        self.save = './kitti_list/'
        
        self.gt = [self.read_groundtruth(i) for i in range(11)]
        self.files = {'train': [], 'val': [], 'test': []}
        
        self.prepare_kitti_ply('test')
        self.prepare_kitti_ply('train')
        self.prepare_kitti_ply('val')
        
        self.refine_poses('test')
        self.refine_poses('train')
        self.refine_poses('val')
        

    def prepare_kitti_ply(self, split='train'):
        for dirname in self.DATA_FILES[split]:
            drive_id = int(dirname)
            fnames = glob.glob(self.root + '/velodyne/sequences/%02d/velodyne/*.bin' % drive_id)
            assert len(fnames) > 0, f"Make sure that the path {self.root} has data {dirname}"
            inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            all_odo = np.genfromtxt(self.root + '/results/%02d.txt' % drive_id)
            all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
            Ts = all_pos[:, :3, 3]
            pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3)) ** 2
            pdist = np.sqrt(pdist.sum(-1))
            more_than_10 = pdist > 10
            curr_time = inames[0]
            while curr_time in inames:
                next_time = np.where(more_than_10[curr_time][curr_time:curr_time + 100])[0]
                if len(next_time) == 0:
                    curr_time += 1
                else:
                    next_time = next_time[0] + curr_time - 1

                if next_time in inames:
                    self.files[split].append((drive_id, curr_time, next_time))
                    curr_time = next_time + 1
        
        if split == 'train':
            self.num_train = len(self.files[split])
            print("Num_train", self.num_train)
        elif split == 'val':
            self.num_val = len(self.files[split])
            print("Num_val", self.num_val)
        else:
            # pair (8, 15, 58) is wrong.
            self.files[split].remove((8, 15, 58))
            self.num_test = len(self.files[split])
            print("Num_test", self.num_test)
    
    def odometry_to_positions(self, odometry):
        T_w_cam0 = odometry.reshape(3, 4)
        T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
        return T_w_cam0
    
    def read_groundtruth(self, seq):
        gt = np.genfromtxt(self.root + '/results/%02d.txt'%seq).reshape([-1, 3, 4])
        gt = np.concatenate([gt,np.repeat(np.array([[[0,0,0,1.]]]),gt.shape[0],axis=0)],axis=1)
        # these transformations are under the left camera's coordinate system
        calibf = open(self.root + '/sequences/%02d/calib.txt'%seq)
        for t in calibf.readlines(): # [P0, P1, P2, P3, Tr]
            if t[0]=='T': t = t[4:]; break # Tr (camera0->LiDAR)
        calibf.close(); calib = np.eye(4, 4)
        calib[:-1, :] = np.array([float(c) for c in t.split(' ')]).reshape([3, 4])
        return np.linalg.inv(calib) @ gt @ calib
    
    def refine_poses(self, split='train'):
        file = open(self.save + split + '.txt', 'w')
        for data_dict in self.files[split]:
            folder = os.path.join(self.root + '/velodyne/sequences/%02d/velodyne/'%data_dict[0])
            src_fn = os.path.join(folder, '%06d.bin'%data_dict[1])
            dst_fn = os.path.join(folder, '%06d.bin'%data_dict[2])
            
            ply1 = o3d.geometry.PointCloud()
            ply2 = o3d.geometry.PointCloud()
            xyz1 = np.fromfile(src_fn, dtype=np.float32, count=-1).reshape([-1,4])[:, :3]
            xyz2 = np.fromfile(dst_fn, dtype=np.float32, count=-1).reshape([-1,4])[:, :3]

            ply1.points = o3d.utility.Vector3dVector(xyz1)
            ply2.points = o3d.utility.Vector3dVector(xyz2)
            ply1 = ply1.voxel_down_sample(voxel_size=0.05)
            ply2 = ply2.voxel_down_sample(voxel_size=0.05)
            
            trans = np.linalg.inv(self.gt[data_dict[0]][data_dict[1]])
            trans = trans @ self.gt[data_dict[0]][data_dict[2]]
            t = o3d.pipelines.registration.registration_icp(
                ply2, ply1, 0.2, trans, # refine the transformation matrix via ICP
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500))
            trans = np.array(t.transformation, dtype=np.float32)[:3].reshape(-1)
            item = '%d %d %d '%(data_dict[0], data_dict[1], data_dict[2])
            item = item + ' '.join(str(k) for k in trans) + '\n'
            file.write(item)
            print(item)
        file.close()


if __name__ == '__main__':
    KITTIDataset()
