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
        self.repeat_instance = 11 // self.input_num
        self.num_theta = cfg["dataset"]["num_atc"]
        self.n_inputs = cfg["dataset"]["num_input_pts"]
        self.inputs_noise_std = cfg["dataset"]["input_noise"] if self.mode == 'train' else 0.
        self.input_type = cfg["dataset"]["input_type"]
        #self.theta_canon = np.array(cfg["dataset"]["theta_canon"], dtype=np.float) * math.pi / 180
        #theta_range
        self.theta_range = np.array(cfg["dataset"]["theta_range"], dtype=np.float) * math.pi / 180

        #file list
        self.meta_list = glob.glob(self.data_root + "/*.mat")
        meta_mode = self.data_root.rsplit('/')[-2]
        meta_category = self.data_root.rsplit('/')[-1]
        meta_rotation_camera = '/'.join(self.data_root.rsplit('/')[:-3]) + '/rotation_camera/'
        #temporary
        if meta_category == 'flip_phone':
            meta_rotation_camera = '/nhome/yuchen_che/rotation_camera/'
        self.rotation_camera = np.load(meta_rotation_camera + meta_mode + '/' + meta_category + '.npy', allow_pickle=True)
        for i in range(len(self.meta_list)):
            dir = self.meta_list[i].split('/')[-1].split('.')[0]
            frame_list = np.random.choice(11, 11, replace=False)
            #camera_list = np.random.rand(11, 3)
            #camera_list  =  2 * camera_list / np.linalg.norm(camera_list, axis=-1, keepdims=True)
            #camera_list[:,1] = np.abs(camera_list[:,1])
            camera_list = self.rotation_camera[i]['camera']
            rotation = self.rotation_camera[i]['rotation']
            self.meta_list[i] = {
                'dir': dir,
                'frame_list': frame_list,
                'camera_list': camera_list,
                'rotation': rotation
            }
        self.meta_list = self.meta_list[: int(len(self.meta_list) * self.dataset_proportion)]
        return

    def __len__(self) -> int:
        return len(self.meta_list) * self.repeat_instance

    def load(self, dir, meta_info, frame):
        instance = sio.loadmat(dir + '/' + meta_info['dir'] + '.mat')['data'][0]
        input = instance['input'][0]
        norm = input[:, :, [3, 5, 4]]
        point = input[:, :, [0, 2, 1]]
        m = instance['m'][0]
        axis = instance['axis'][0]
        axis_t = axis[:, [0, 2, 1]]
        axis_o = axis[:, [3, 5, 4]]
        T = instance['T'][0]
        label = instance['label'][0].astype(np.int)
        hpr_idx_list = []
        rotation = meta_info['rotation']

        # select frames
        if self.input_num < T.item():
            #idx_t = meta_info['frame_list'][frame * self.input_num: (frame+1) * self.input_num]
            idx_t = meta_info['frame_list'][np.random.choice(11, self.input_num, replace=False)]
            point = point[:, idx_t, :]
            norm = norm[:, idx_t, :]
        else:
            idx_t = meta_info['frame_list']
        # dep
        if self.input_type == 'dep':
            camera_list = meta_info['camera_list'][idx_t, :]
            for t in range(self.input_num):
                frame_pc = o3d.geometry.PointCloud()
                frame_pc.points = o3d.utility.Vector3dVector(point[:, t, :])
                camera = camera_list[t]
                _, hpr_idx = frame_pc.hidden_point_removal(camera, radius=1000)
                #logging.warning("hpr_len:{}".format(len(hpr_idx)))
                replace_idx = len(hpr_idx) < self.n_inputs
                selected_idx = np.random.choice(len(hpr_idx), self.n_inputs, replace=replace_idx)
                hpr_idx = np.array(hpr_idx)[selected_idx]
                hpr_idx_list.append(hpr_idx)
            hpr_idx_list = np.stack(hpr_idx_list, axis=0)
            hpr_idx_mask = hpr_idx_list.T[..., np.newaxis].repeat(repeats=3, axis=-1)
            point = np.take_along_axis(point, indices=hpr_idx_mask, axis=0)
            label = label[:, np.newaxis, :].repeat(repeats=self.input_num, axis=1)
            label = np.take_along_axis(label, indices=hpr_idx_mask[:, :, :1], axis=0)
        # pcl
        else:
            if self.n_inputs < point.shape[0]:
                idx_p = np.random.choice(point.shape[0], size=self.n_inputs, replace=False)
                point = point[idx_p, :, :]
                norm = norm[idx_p, :, :]
                label = label[idx_p, np.newaxis, :].repeat(self.input_num, axis=1)
            else:
                label = label[:, np.newaxis, :].repeat(self.input_num, axis=1)
        
        return point, norm, m, axis_t, axis_o, label, hpr_idx_list, rotation

    def __getitem__(self, index: int):
        ret = {}
        meta_info = self.meta_list[index // self.repeat_instance]
        meta_info["viz_id"] = f"{self.mode}_{os.path.basename(meta_info['dir'])}_idx{index}"
        meta_info["mode"] = self.mode
        meta_info['category'] = self.data_root.split('/')[-1]
        repeat_frame = index % self.repeat_instance
        ret["object_T"] = np.eye(4)

        #load
        point, norm, m, axis_t, axis_o, label, hpr_list, rotation_vec = self.load(
            self.data_root, meta_info, repeat_frame)
        
        #scale
        if self.scale:
            scale_parameter = 0.5 * np.random.rand(1) + 0.75
            point *= scale_parameter
            axis_t *= scale_parameter

        #rotation
        if self.rotate:
            #rand_vec = 2 * torch.rand(1, 3) - 1
            #rand_vec[:,0] = 0
            #rand_vec[:,2] = 0
            #rand_vec = ExpSO3(math.pi * rand_vec).unsqueeze(1).numpy()#1,1,3,3
            rand_vec = rotation_vec
            point = (rand_vec @ point[..., np.newaxis]).squeeze(-1)
            norm = (rand_vec @ norm[..., np.newaxis]).squeeze(-1)
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
        ret["norm"] = norm.transpose(1,0,2)
        #ret['m'] = m
        ret['axis_o'] = axis_o
        ret['axis_t'] = axis_t
        ret['label'] = label
        #ret['theta_canon'] = self.theta_canon
        ret['theta_range'] = self.theta_range
        if self.input_type == 'dep':
            ret['hpr_list'] = hpr_list
 
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