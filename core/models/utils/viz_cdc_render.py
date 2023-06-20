# use pyrender to viz cdc
import numpy as np
from core.models.utils.pyrender_helper import render
from copy import deepcopy
from matplotlib import cm


def align(p0, p1):
    # R rotate p1 tp p0
    x0 = deepcopy(p0)
    y0 = deepcopy(p1)
    q0 = x0 - x0.mean(axis=0)
    q1 = y0 - y0.mean(axis=0)
    W = (q1[..., None] @ q0[:, None, :]).sum(axis=0)
    U, s, VT = np.linalg.svd(W)
    R = VT.T @ U.T
    return R


def viz_cdc(
    input_pc,
    object_T=None,
    scale_cdc=True,
    interval=1,
    align_cdc=False,
    cam_dst_default=1.0,
    input_pc_mask=None,
    input_pc_joint=None,
):
    # prepare viz
    T = input_pc.shape[0]

    if object_T is None:
        object_T = np.eye(4)
    viz_pc_list, viz_pc_mask_list = [], []

    inv_T = np.linalg.inv(object_T)
    for t in range(T):
        pc = input_pc[t]  # N,3
        viz_pc_list.append((inv_T[:3, :3] @ pc.T + inv_T[:3, 3:4]).T)

        if isinstance(input_pc_mask, np.ndarray):
            viz_pc_mask_list.append(input_pc_mask[t])

    # align cdc to the pose of posed frame first frame
    object_T = np.eye(4)
    query_viz_list = []
    fig_t_list = []
    t_color = cm.summer(np.array([float(i) / float(T) for i in range(T)]))[:, :3]

    for t in range(T):
        if t % interval != 0:
            continue
        # render input pc
        rgb_input, _ = render(
            point_cloud=viz_pc_list[t],
            point_cloud_r=0.008,
            point_cloud_color=viz_pc_mask_list[t] if viz_pc_mask_list != [] else None,
            point_cloud_material_color=t_color[t],  # [0.0, 0.39, 1.0],
            light_intensity=4.0, 
            cam_dst_default=cam_dst_default,
            joint=input_pc_joint
        )
        fig_t_list.append(rgb_input)
    return fig_t_list, query_viz_list
