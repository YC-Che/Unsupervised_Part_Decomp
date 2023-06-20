# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

import torch
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds

def _validate_chamfer_reduction_inputs(
    batch_reduction: Union[str, None], point_reduction: str
) -> None:
    """Check the requested reductions are valid.

    Args:
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
    """
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
    if point_reduction not in ["mean", "sum"]:
        raise ValueError('point_reduction must be one of ["mean", "sum"]')


def _handle_pointcloud_input(
    points: Union[torch.Tensor, Pointclouds],
    lengths: Union[torch.Tensor, None],
    normals: Union[torch.Tensor, None],
):
    """
    If points is an instance of Pointclouds, retrieve the padded points tensor
    along with the number of points per batch and the padded normals.
    Otherwise, return the input points (and normals) with the number of points per cloud
    set to the size of the second dimension of `points`.
    """
    if isinstance(points, Pointclouds):
        X = points.points_padded()
        lengths = points.num_points_per_cloud()
        normals = points.normals_padded()  # either a tensor or None
    elif torch.is_tensor(points):
        if points.ndim != 3:
            raise ValueError("Expected points to be of shape (N, P, D)")
        X = points
        if lengths is not None:
            if lengths.ndim != 1 or lengths.shape[0] != X.shape[0]:
                raise ValueError("Expected lengths to be of shape (N,)")
            if lengths.max() > X.shape[1]:
                raise ValueError("A length value was too long")
        if lengths is None:
            lengths = torch.full(
                (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
            )
        if normals is not None and normals.ndim != 3:
            raise ValueError("Expected normals to be of shape (N, P, 3")
    else:
        raise ValueError(
            "The input pointclouds should be either "
            + "Pointclouds objects or torch.Tensor of shape "
            + "(minibatch, num_points, 3)."
        )
    return X, lengths, normals


def classification_chamfer(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    weights=None,
    batch_reduction = "mean",
    point_reduction = "mean",
    norm: int = 2,
    pcl = False,
    dcd_alpha = None,
):
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
        torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]
    '''
    trans = torch.cat([c_trans[:,:-1], c_trans[:,:1]],dim=1).detach()
    axis = torch.cat([c_axis[:,:-1], c_axis[:,:1]],dim=1).detach()
    x_1 = x.reshape(B,P,T,-1,3) - trans.unsqueeze(-2).unsqueeze(-2)#B,P,T,N,3
    x_axis_projection = torch.sum(x_1 * axis.unsqueeze(-2).unsqueeze(-2), dim=-1).unsqueeze(-1)#B,P,T,N,1
    x_axis_projection = axis.unsqueeze(-2).unsqueeze(-2) * x_axis_projection.expand(-1,-1,-1,-1,3)#B,P,T,N,3
    x_1 = (x_1 - x_axis_projection).reshape(B*P*T,-1,3)
    y_1 = y.reshape(B,P,T,-1,3) - trans.unsqueeze(-2).unsqueeze(-2)#B,P,T,N,3
    y_axis_projection = torch.sum(y_1 * axis.unsqueeze(-2).unsqueeze(-2), dim=-1).unsqueeze(-1)#B,P,T,N,1
    y_axis_projection = axis.unsqueeze(-2).unsqueeze(-2) * y_axis_projection.expand(-1,-1,-1,-1,3)#B,P,T,N,3
    y_1 = (y_1 - y_axis_projection).reshape(B*P*T,-1,3)
    x_nn_1 = knn_points(x_1, y_1, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)
    cham_x_1 = x_nn_1.dists[..., 0]  # (N, P1)
    cham_x_1 = 1 - torch.exp(-dcd_alpha * cham_x_1)
    '''
    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)
    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_x = 1 - torch.exp(-dcd_alpha * cham_x)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if weights is not None:
        cham_x *= weights

    # Apply point reduction
    # if pcl:
    #     cham_x = cham_x.sum(1) + cham_x_1.sum(1)  # (N,)
    # else:
    cham_x = cham_x.sum(1)

    if point_reduction == "mean":
        x_lengths_clamped = x_lengths.clamp(min=1)
        cham_x /= x_lengths_clamped
        cham_x /= N

    return cham_x.sum()


def align_chamfer(
    x,  # BPT,N,3
    y,#BPT,(T-1)N,3
    B,
    P,
    T,
    c_trans,
    c_axis,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    weights=None,
    batch_reduction="mean",
    point_reduction="mean",
    norm: int = 2,
    threshold_weight=1,
    dcd_alpha=None
):
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)
    N = x.shape[1]

    origin_x = x.reshape(B,P,T,-1,3)[:,-1,:,:,:]#B,T,N,3
    origin_y = y.reshape(B,P,T,-1,3)[:,-1,:,:,:]#B,T,(T-1)N,3
    static_knn = knn_points(origin_x.reshape(B*T,-1,3), origin_y.reshape(B*T,-1,3), lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)
    static_cham = static_knn.dists[..., 0].reshape(B,-1)#B, TN

    #threshold = torch.quantile(static_cham, q=0.8, dim=-1)
    threshold = torch.mean(static_cham, dim=-1)#B
    threshold *= threshold_weight
    active_mask = static_cham >= threshold.unsqueeze(-1)
    active_mask = active_mask.reshape(B,T,-1) #B,T,N
    if weights!=None:
        active_mask = ~weights.bool()
    static_mask = ~active_mask

    #torch.save(origin_x.reshape(B,-1,3), './chamfer_x.pt')
    #torch.save(static_cham, './chamfer_c.pt')

    active_x = []
    for b in range(B):
        for p in range(P-1):
            for t in range(T):
                active = x.reshape(B,P,T,-1,3)[b,p,t][active_mask[b,t],:]
                if active.shape[0] == 0:
                    active_x.append(10 * torch.ones((N,3), device=x.device, dtype=x.dtype))#N,3
                else:
                    joint_trans = c_trans[b,p]
                    joint_axis = c_axis[b,p]
                    active_cham = torch.norm(active - joint_trans.unsqueeze(0), dim=-1, keepdim=True)
                    active_x.append(torch.cat([active, 10 * torch.ones((N-active.shape[0], 3), device=x.device, dtype=x.dtype)]))#N,3
    active_x = torch.stack(active_x, dim=0)
    active_x = active_x.reshape(B,P-1,T,-1,3)#B,P-1,T,N,3

    exclude = [[list(range(i)), list(range(i+1, T))]for i in range(T)]
    exclude = [exclude[i][0] + exclude[i][1] for i in range(T)] #T,T-1
    active_y = torch.stack([active_x[:,:,exclude[i],:,:] for i in range(T)], dim=2)#B,P-1,T,T-1,N,4
    active_x = active_x.reshape(B*(P-1)*T,-1,3)
    active_y = active_y.reshape(B*(P-1)*T,-1,3)
    active_cham = knn_points(active_x, active_y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)
    active_cham = active_cham.dists[..., 0].reshape(B,P-1,-1)#B,P-1,TN

    active_cham = 1 - torch.exp(-dcd_alpha * active_cham)
    if P>2:
        active_cham = torch.min(active_cham, dim=1).values #B,TN

    return active_cham.flatten().mean(), static_mask, active_mask

def static_suppreession(
    x,
    points,
    points_occ,
    B,
    P,
    T,
    weights=None,
    norm: int = 2,
):
    '''
    x: BPT,N,3
    points:B,M,3
    points_occ:B,M
    label:B,P,T,N
    weight:B,P,T,N
    '''

    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")
    N = x.shape[1]

    surface = x.reshape(B,P,-1,3)[:,-1]#B,TN,3
    outside = torch.where(points_occ.unsqueeze(-1).expand(-1,-1,3)==0, points, 10*torch.ones_like(points))
    inside = torch.where(points_occ.unsqueeze(-1).expand(-1,-1,3)==1, points, 10*torch.ones_like(points))

    outside_nn = knn_points(surface, outside, norm=norm, K=1)
    cham_outside = outside_nn.dists[..., 0]
    cham_outside = cham_outside.reshape(B,T,-1)
    inside_nn = knn_points(surface, inside, norm=norm, K=1)
    cham_inside = inside_nn.dists[..., 0]
    cham_inside = cham_inside.reshape(B,T,-1)

    cham = cham_inside + cham_outside#B,T,N
    active_threshold = 2 * torch.mean(cham.reshape(B,-1), dim=1)#B
    active_mask = cham >= active_threshold.unsqueeze(-1).unsqueeze(-1)#B,T,N
    static_weight = weights[:,-1]#B,T,N
    static_weight_active_point = static_weight.flatten()[active_mask.flatten()]

    loss = torch.maximum(static_weight_active_point - 0.5, torch.zeros_like(static_weight_active_point)).mean()
    return loss, active_mask


if __name__ == '__main__':
    x = torch.randn((6*2*4,500,3)).cuda()
    y = torch.randn((6*2*4,1500,3)).cuda()
    print(align_chamfer(x,y,6,2,4))
