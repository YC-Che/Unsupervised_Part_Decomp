# new dataset for dt4d dataset
import torch
from torch.utils.data import Dataset
import os
import numpy as np
from os.path import join
import math
import scipy.io as sio
import glob
import open3d as o3d
import logging

class Dataset(Dataset):
    def __init__(self, cfg, mode) -> None:
        super().__init__()
        # init
        self.mode = mode.lower()
        self.rotate = cfg["dataset"]["rotate_augmentation"]
        self.translate = cfg["dataset"]["translate_augmentation"] if self.mode == 'train' else False
        self.scale = cfg["dataset"]["scale_augmentation"] if self.mode == 'train' else False
        self.noise = cfg["dataset"]["input_noise"] > 0 and self.mode == 'train'
        self.dataset_proportion = cfg["dataset"]["dataset_proportion"][cfg["modes"].index(self.mode)]
        self.data_root = join(cfg["root"], cfg["dataset"]["data_root"])
        self.input_num = cfg["dataset"]["input_num"]
        self.repeat_instance = 2
        self.num_theta = cfg["dataset"]["num_atc"]
        self.n_inputs = cfg["dataset"]["num_input_pts"]
        self.inputs_noise_std = cfg["dataset"]["input_noise"] if self.mode == 'train' else 0.
        self.input_type = cfg["dataset"]["input_type"]
        #self.theta_canon = np.array(cfg["dataset"]["theta_canon"], dtype=np.float) * math.pi / 180
        #theta_range
        self.theta_range = np.array(cfg["dataset"]["theta_range"], dtype=np.float) * math.pi / 180

        #file list
        self.meta_list = glob.glob(self.data_root + "/**/*.npz", recursive=True)
        for i in range(len(self.meta_list)):
            dir = '/'.join(
                [self.meta_list[i].rsplit('/')[-3], self.meta_list[i].rsplit('/')[-2], self.meta_list[i].rsplit('/')[-1].split('.')[0]]
                )
            self.meta_list[i] = {
                'dir': dir,
            }
        self.meta_list = self.meta_list[: int(len(self.meta_list) * self.dataset_proportion)]
        return

    def __len__(self) -> int:
        return len(self.meta_list)

    def load(self, dir, meta_info, frame):
        instance = np.load(dir + '/' + meta_info['dir'] + '.npz', allow_pickle=True)
        #if frame == 0:
        #    frame_0, frame_1, label = 'pc_start', 'pc_start_end', 'pc_seg_start'
        #else:
        #    frame_0, frame_1, label = 'pc_end', 'pc_end_start', 'pc_seg_end'
        frame_0, frame_1, label_0, label_1 = 'pc_start', 'pc_end', 'pc_seg_start','pc_seg_end'

        point_0 = instance[frame_0][:, np.newaxis,:][:, :, [0, 2, 1]]
        point_1 = instance[frame_1][:, np.newaxis,:][:, :, [0, 2, 1]]

        axis_t = np.cross(instance['screw_axis'], instance['screw_moment'])[np.newaxis, :]
        axis_t = axis_t[:, [0,2,1]]
        axis_o = instance['screw_axis'][np.newaxis, :]
        axis_o = axis_o[:, [0,2,1]]

        #label = instance[label].astype(np.int)[:, np.newaxis, np.newaxis].repeat(self.input_num, axis=1)
        label_0 = instance[label_0].astype(np.int)[:, np.newaxis, np.newaxis]
        label_1 = instance[label_1].astype(np.int)[:, np.newaxis, np.newaxis]

        #scale = (np.maximum(point_0.max(0), point_1.max(0)) - np.minimum(point_0.min(0), point_1.min(0))).max()
        #scale *= 1.1
        #scale = instance['start_mesh_pose_dict'].item()['0_0'][0][1][0]
        #point_0 /= scale
        #point_1 /= scale
        #axis_t /= scale

        if self.n_inputs != point_0.shape[0]:
            if self.n_inputs < point_0.shape[0]:
                idx_p = np.random.choice(point_0.shape[0], size=self.n_inputs, replace=False)
            else:
                idx_p = np.random.choice(point_0.shape[0], size=self.n_inputs, replace=True)
            point_0 = point_0[idx_p, :, :]
            label_0 = label_0[idx_p, :, :]

        if self.n_inputs != point_1.shape[0]:
            if self.n_inputs < point_1.shape[0]:
                idx_p = np.random.choice(point_1.shape[0], size=self.n_inputs, replace=False)
            else:
                idx_p = np.random.choice(point_1.shape[0], size=self.n_inputs, replace=True)
            point_1 = point_1[idx_p, :, :]
            label_1 = label_1[idx_p, :, :]
        
        label = np.concatenate([label_0, label_1], axis=1)
        point = np.concatenate([point_0, point_1], axis=1)
        return point, axis_t, axis_o, label

    def __getitem__(self, index: int):
        ret = {}
        meta_info = self.meta_list[index]
        repeat_frame = index % self.repeat_instance
        meta_info["viz_id"] = f"{self.mode}_{meta_info['dir'].split('_')[0]}_{meta_info['dir'].split('/')[-1]}_idx{repeat_frame}"
        meta_info["mode"] = self.mode
        meta_info['category'] = meta_info['dir'].split('_')[0]
        ret["object_T"] = np.eye(4)

        #load
        point, axis_t, axis_o, label = self.load(
            self.data_root, meta_info, repeat_frame)
        
        #scale
        scale = (point.reshape(-1, 3).max(0) - point.reshape(-1, 3).min(0)).max()
        scale *= 1.1
        point /= scale
        axis_t /= scale
        if self.scale:
            scale_parameter = 0.5 * np.random.rand(1) + 0.75
            point *= scale_parameter
            axis_t *= scale_parameter

        #rotation
        if self.rotate:
            rand_vec = 2 * torch.rand(1, 3) - 1
            rand_vec[:,0] = 0
            rand_vec[:,2] = 0
            rand_vec = ExpSO3(math.pi * rand_vec).unsqueeze(1).numpy()#1,1,3,3
            point = (rand_vec @ point[..., np.newaxis]).squeeze(-1)
            #norm = (rand_vec @ norm[..., np.newaxis]).squeeze(-1)
            axis_o = (rand_vec[:,0] @ axis_o[..., np.newaxis]).squeeze(-1)
            axis_t = (rand_vec[:,0] @ axis_t[..., np.newaxis]).squeeze(-1)

        #normalization
        normalization_t = np.mean(point.reshape(-1,3), axis=0)
        point -= normalization_t.reshape(1,1,3)
        axis_t -= normalization_t.reshape(1, 3)
        
        #noise
        if self.noise:
            noise = self.inputs_noise_std * np.random.randn(*point.shape)
            noise = noise.astype(np.float32)
            point = noise + point
        
        #translation
        if self.translate:
            rand_trans = 0.5 * np.random.rand(3) - 0.25
            point += rand_trans.reshape(1,1,3)
            axis_t += rand_trans.reshape(1,3)

        #output
        ret["inputs"] = point.transpose(1,0,2)
        #ret["norm"] = norm.transpose(1,0,2)
        #ret['m'] = m
        ret['axis_o'] = axis_o
        ret['axis_t'] = axis_t
        ret['label'] = label
        #ret['theta_canon'] = self.theta_canon
        ret['theta_range'] = self.theta_range
 
        return ret, meta_info


    #https://albert.growi.cloud/627ceb255286c3b02691384d
def hat(phi):
    phi_x = phi[..., 0]
    phi_y = phi[..., 1]
    phi_z = phi[..., 2]
    zeros = torch.zeros_like(phi_x)

    phi_hat = torch.stack([
        torch.stack([ zeros, -phi_z,  phi_y], dim=-1),
        torch.stack([ phi_z,  zeros, -phi_x], dim=-1),
        torch.stack([-phi_y,  phi_x,  zeros], dim=-1)
    ], dim=-2)
    return phi_hat


def ExpSO3(phi, eps=1e-4):
    theta = torch.norm(phi, dim=-1)
    phi_hat = hat(phi)
    I = torch.eye(3, device=phi.device)
    coef1 = torch.zeros_like(theta)
    coef2 = torch.zeros_like(theta)

    ind = theta < eps

    # strict
    _theta = theta[~ind]
    coef1[~ind] = torch.sin(_theta) / _theta
    coef2[~ind] = (1 - torch.cos(_theta)) / _theta**2

    # approximate
    _theta = theta[ind]
    _theta2 = _theta**2
    _theta4 = _theta**4
    coef1[ind] = 1 - _theta2/6 + _theta4/120
    coef2[ind] = .5 - _theta2/24 + _theta4/720

    coef1 = coef1[..., None, None]
    coef2 = coef2[..., None, None]
    return I + coef1 * phi_hat + coef2 * phi_hat @ phi_hat