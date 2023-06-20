# some utils for oflow

import numpy as np
import math
import logging
from core.models.utils.align import rotation_vec_2_matrix, multi_frame_align
from pytorch3d.loss import chamfer_distance
import torch
import torch.nn.functional as F

def eval_segmentation_acc(x_hat, x):
    '''
    x_hat: B,N,P
    x:B,N,1
    '''
    if x_hat.ndim == 2:
        x_hat = x_hat[np.newaxis, ...]
        x = x[np.newaxis, ...]
    B,N,_ = x_hat.shape
    acc_list = []
    parts_list = []
    for b in range(B):
        n_parts = x.flatten().max() + 1

        if n_parts == 2:
            parts_0_idx = x[b,:,0]==0
            parts_1_idx = x[b,:,0]==1
            tmp0 = np.count_nonzero(x_hat[b, parts_0_idx, 1]) + np.count_nonzero(x_hat[b, parts_1_idx, 0])
            tmp1 = np.count_nonzero(x_hat[b, parts_0_idx, 0]) + np.count_nonzero(x_hat[b, parts_1_idx, 1])
            if tmp0>tmp1:
                acc = tmp0
                parts_indice = [1, 0]
            else:
                acc = tmp1
                parts_indice = [0, 1]
        elif n_parts == 3:
            parts_0_idx = x[b,:,0]==0
            parts_1_idx = x[b,:,0]==1
            parts_2_idx = x[b,:,0]==2
            acc = 0
            for i, j, k in [[0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]]:
                tmp = np.count_nonzero(x_hat[b, parts_0_idx, i])
                tmp += np.count_nonzero(x_hat[b, parts_1_idx, j])
                tmp += np.count_nonzero(x_hat[b, parts_2_idx, k])
                if tmp > acc:
                    acc = tmp
                    parts_indice = [i, j, k]
        else:
            # とりあえず3partsまで
            acc = 0
            logging.warning("acc fail!")
            logging.warning(n_parts)
        acc /= N
        acc_list.append(acc)
        parts_list.append(parts_indice)
    
    acc_list = np.array(acc_list)
    acc_mean = acc_list.mean()
    return acc_list, acc_mean, parts_list

def eval_segmentation_iou(x_hat, x, parts_indice, motion_part=True):
    '''
    x_hat: B,N,P
    x:B,N,1
    pi:B,P
    '''
    if x_hat.ndim == 2:
        x_hat = x_hat[np.newaxis, ...]
        x = x[np.newaxis, ...]
    B,N,P = x_hat.shape
    iou_list = []

    for b in range(B):
        n_parts = x.flatten().max() + 1

        if n_parts == 2:
            if parts_indice[b] == [0,1]:
                parts_0_idx = x[b,:,0]==0
            else:
                parts_0_idx = x[b,:,0]==1
            intersection = np.count_nonzero(x_hat[b, parts_0_idx, 0])
            union = np.count_nonzero(x_hat[b, :, 0]) + np.count_nonzero(parts_0_idx) - intersection
            iou_mean = intersection / union if union != 0 else 1
            if not motion_part:
                static_union = N - intersection
                static_intersection = N - union
                iou_mean += static_intersection / static_union
                iou_mean /= 2

        elif n_parts == 3:
            parts_0_idx = x[b,:,0]==0
            parts_1_idx = x[b,:,0]==1
            parts_2_idx = x[b,:,0]==2
            acc = 0
            i,j,k = parts_indice[b]
            intersection_i = np.count_nonzero(x_hat[b, parts_0_idx, i])
            union_i = np.count_nonzero(x_hat[b, :, i]) + np.count_nonzero(parts_0_idx) - intersection_i
            iou_i = intersection_i / union_i if np.count_nonzero(parts_0_idx) != 0 else 1
            intersection_j = np.count_nonzero(x_hat[b, parts_1_idx, j])
            union_j = np.count_nonzero(x_hat[b, :, j]) + np.count_nonzero(parts_1_idx) - intersection_j
            iou_j = intersection_j / union_j if np.count_nonzero(parts_1_idx) != 0 else 1
            intersection_k = np.count_nonzero(x_hat[b, parts_2_idx, k])
            union_k = np.count_nonzero(x_hat[b, :, k]) + np.count_nonzero(parts_2_idx) - intersection_k
            iou_k = intersection_k / union_k  if np.count_nonzero(parts_2_idx) != 0 else 1
            if not motion_part:
                if i == 2:
                    iou_mean = 0.5 * (iou_j + iou_k)
                if j == 2:
                    iou_mean = 0.5 * (iou_i + iou_k)
                if k == 2:
                    iou_mean = 0.5 * (iou_i + iou_j)
            else:
                iou_mean = 1/3 * (iou_i + iou_j + iou_k)
        else:
            # とりあえず3partsまで
            logging.warning("iou fail!")
            iou_mean = 0
        iou_list.append(iou_mean)
    
    iou_list = np.array(iou_list)
    iou_mean = iou_list.mean()
    return iou_list, iou_mean

def eval_orientation_error(o_hat, o):
    '''
    o_hat:B,P-1,3
    o:B,P-1,3
    '''
    if o_hat.ndim == 2:
        o_hat = o_hat[np.newaxis,...]
        o = o[np.newaxis,...]
    B,A,_ = o.shape
    err_list = []
    for b in range(B):
        hat = o_hat[b]
        gt = o[b]
        hat /= (np.sum(hat**2, axis=-1, keepdims=True) ** 0.5 + 1e-5)
        gt /= (np.sum(gt**2, axis=-1, keepdims=True) ** 0.5 + 1e-5)
        if A == 1:
            err = np.arccos(np.abs(np.sum(hat * gt, axis=-1)))
            err = np.mean(err, axis=-1)
        else:
            tmp1 = np.abs(np.sum(hat * gt, axis=-1))
            tmp1 = np.mean(np.arccos(tmp1))
            tmp2 = np.abs(np.sum(hat[[1,0],:] * gt, axis=-1))
            tmp2 = np.mean(np.arccos(tmp2))
            err = tmp1 if tmp1 < tmp2 else tmp2

        err = err * 180 / math.pi
        err_list.append(err)
    
    err_list = np.array(err_list)
    err_mean = err_list.mean()
    return err_list, err_mean

def eval_min_distance(t_hat, t, o):
    '''
    t_hat: B,P-1,3
    t: B,P-1,3
    o:B,P-1,3
    '''
    if t_hat.ndim == 2:
        t_hat = t_hat[np.newaxis,...]
        t = t[np.newaxis,...]
        o = o[np.newaxis,...]
    B,A,_ = o.shape
    distance_list = []
    for b in range(B):
        if A == 1:
            t_diff = t_hat[b] - t[b]
            o_line = o[b] / (np.sum(o[b]**2, axis=-1, keepdims=True) ** 0.5)
            projection = np.abs(np.sum(t_diff * o_line, axis=-1))
            distance = (np.sum(t_diff ** 2, axis=-1) - projection ** 2) ** 0.5
        else:
            t_diff = t_hat[b] - t[b]
            o_line = o[b] / (np.sum(o[b]**2, axis=-1, keepdims=True) ** 0.5)
            projection = np.abs(np.sum(t_diff * o_line, axis=-1))
            tmp_1= np.mean((np.sum(t_diff ** 2, axis=-1) - projection ** 2) ** 0.5)
            t_diff = t_hat[b,[1,0]] - t[b]
            o_line = o[b,[1,0]] / (np.sum(o[b]**2, axis=-1, keepdims=True) ** 0.5)
            projection = np.abs(np.sum(t_diff * o_line, axis=-1))
            tmp_2= np.mean((np.sum(t_diff ** 2, axis=-1) - projection ** 2) ** 0.5)
            distance = tmp_1 if tmp_1 < tmp_2 else tmp_2
        distance_list.append(distance)
    
    distance_list = np.array(distance_list)
    distance_mean = distance.mean()
    return distance_list, distance_mean

def eval_epe(pc_gt, segmentation_gt, parts_indices, joint_t_pred, theta_pred):
    '''
    gt_pc: T, N, 3
    segmentation_gt: T, N, P
    parts_indice: P
    joint_t_pred: P, 3
    theta_pred: P, T, 3, 3
    '''
    T,N,_ = pc_gt.shape
    rotations = theta_pred#P,T,3,3
    joint_t_pred = torch.tensor(joint_t_pred)
    if len(segmentation_gt.shape)==2:
        segmentation_gt = segmentation_gt[np.newaxis, ...]
    segmentation_gt = torch.tensor(segmentation_gt)[0]#N,1
    segmentation_gt = F.one_hot(segmentation_gt.squeeze(-1).long())

    if joint_t_pred.shape[0] != len(parts_indices):
        joint_t_pred = torch.cat([joint_t_pred, torch.zeros_like(joint_t_pred[-1]).unsqueeze(0)], axis=0)
        #rotations = torch.cat([torch.eye(3).reshape(1,1,3,3).expand(1,T,3,3), theta_pred], axis=0)

    distance_list = []
    for i in range(2):
        if i == 0:
            aligned_pc = multi_frame_align(
                pc_gt.unsqueeze(0),
                rotations.unsqueeze(0),
                joint_t_pred.unsqueeze(0),
                cat_frame_label=False)#B,P,T,N,3
        else:
            aligned_pc = multi_frame_align(
                pc_gt.unsqueeze(0),
                torch.transpose(rotations,-1,-2).unsqueeze(0),
                joint_t_pred.unsqueeze(0),
                cat_frame_label=False)#B,P,T,N,3
        if segmentation_gt.shape[-1] == 3:
            comb_list = [[0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]]
        elif segmentation_gt.shape[-1] == 2:
            comb_list = [[0,1], [1,0]]
        for comb in comb_list:
            segmentation_gt_tmp = segmentation_gt[:, comb]
            test = aligned_pc.squeeze(0) * segmentation_gt_tmp.permute(1,0).unsqueeze(1).unsqueeze(-1)
            test = torch.sum(test, dim=0)#T,N,3
            #np.savez('aligned_pc_1.npz', aligned_pc.numpy()) if i==0 else np.savez('aligned_pc_2.npz', aligned_pc.numpy())
            distance = torch.sum((test[0].unsqueeze(0) - test[1:]) ** 2, dim=-1) ** 0.5
            distance = distance.detach().flatten().mean().numpy()
            distance_list.append(distance)

    return min(distance_list)

            


def eval_atc_all(joint_o_pred, joint_t_pred, segmentation_pred, joint_o_gt, joint_t_gt, segmentation_gt, pc=None, rotation=None):
    '''
    joint_o_pred: P-1,3
    joint_t_pred: P-1,3
    segmentation_pred: N,P
    joint_o_gt: P-1,3
    joint_t_gt: P-1,3
    segmentation_gt: N,1
    pc_gt : B,T,N,3
    '''
    seg_ret, _, parts_indice = eval_segmentation_acc(x_hat=segmentation_pred, x=segmentation_gt)
    motion_iou_ret, _ = eval_segmentation_iou(segmentation_pred, segmentation_gt, parts_indice, motion_part=True)
    overall_iou_ret, _ = eval_segmentation_iou(segmentation_pred, segmentation_gt, parts_indice, motion_part=False)
    seg_ret = seg_ret.mean()
    motion_iou_ret = motion_iou_ret.mean()
    overall_iou_ret = overall_iou_ret.mean()
    ori_ret, _ = eval_orientation_error(joint_o_pred, joint_o_gt)
    dis_ret, _ = eval_min_distance(joint_t_pred, joint_t_gt, joint_o_gt)

    ret_dict = {
        'segmentation_mean':seg_ret,
        'motion_part_iou': motion_iou_ret,
        'overall_iou': overall_iou_ret,
        'joint_orientation_error_mean':ori_ret,
        'joint_distance_error':dis_ret
        }
    
    if pc != None:
        epe_ret = eval_epe(pc, segmentation_gt, parts_indice[0], joint_t_pred, rotation)
        ret_dict['end_point_error'] = epe_ret
    return ret_dict
