import torch

def supervised_axis_loss(
        c_axis,
        axis_o
    ):
    axis_pred = c_axis[:, :-1, :]
    #trans_pred = c_trans[:, :-1, :]
    loss = 1 - torch.abs(torch.sum(axis_pred * axis_o, dim=-1))
    #loss += torch.sum((trans_pred - axis_t)**2, dim=-1) ** 0.5
    return loss.flatten().mean()

def theta_range_loss(
        c_length,
        theta_range
    ):
    # range:B,P-1,2
    # c_length:B,P-1,T
    B, P, T = c_length.shape
    P += 1
    x = c_length.permute(0, 2, 1).reshape(-1, P-1)  # BT, P-1
    range = theta_range.unsqueeze(1).expand(-1, T, -1, -1).reshape(x.shape[0], -1, 2)  # BT,P-1,2
    min_dis = torch.maximum(range[:, :, 0] - x, torch.zeros_like(x)).flatten()
    max_dis = torch.maximum(x - range[:, :, 1], torch.zeros_like(x)).flatten()
    y = torch.mean(min_dis + max_dis, dim=-1)
    return y

def joint_decoder_loss(
        axis,
        length,
        trans,
        set_pc,
        mask,
        refine_threshold=100,
        ablation=False
    ):
    '''
    axis: B,T,P,3
    trans: B,T,P,3
    length: B,P,T
    theta_gt: B,T,P-1
    trans:B,P,3
    set_pc: B,T,N,3
    mask:B,P,T,N
    '''
    B, P, T = length.shape

    # rotation axis norm remain 1
    axis_length = torch.norm(axis.reshape(-1, 3), dim=1)
    norm_loss = torch.mean((axis_length - 1) ** 2)

    if trans.ndim == 4:
        # location variance for each joint in different frame
        variance = torch.var(trans[:,:,:-1,:], unbiased=False, dim=1)#B,P,3
        # direction variance for each joint in different frame
        variance += torch.var(axis[:,:,:-1,:], unbiased=False, dim=1)#B,P,3
        variance_loss = torch.mean(variance.flatten())
    else:
        variance_loss = 0

    # trans should be closed to point cloud
    trans_loss = []
    for b in range(B):
        for t in range(set_pc.shape[1]):
            frame_mask = mask[b, :, t, :]  # P,N
            parts_existance = torch.count_nonzero(frame_mask, dim=-1)  # p
            joint = trans[b, :-1] if trans.ndim == 3 else trans[b, t, :-1]
            # refinement with each parts
            if torch.all(parts_existance >= refine_threshold) and not ablation:
                active_pc_0 = set_pc[b,t].reshape(-1,3)[frame_mask[0, :] == 1, :]
                if P == 3:
                    active_pc_1 = set_pc[b,t].reshape(-1,3)[frame_mask[1, :] == 1, :]
                static_pc = set_pc[b,t].reshape(-1,3)[frame_mask[-1, :] == 1, :]

                active_distance = torch.min(
                    torch.sum(
                        (joint[:1,:].unsqueeze(1) - active_pc_0.unsqueeze(0).expand(P-1,-1,-1))**2,
                        dim=-1),
                    dim=-1).values
                if P == 3:
                    active_distance += torch.min(
                        torch.sum(
                            (joint[1:2,:].unsqueeze(1) - active_pc_1.unsqueeze(0).expand(P-1,-1,-1))**2,
                            dim=-1),
                        dim=-1).values
                static_distance = torch.sum((joint.unsqueeze(1) - static_pc.unsqueeze(0).expand(P-1,-1,-1))**2, dim=-1)#P-1,M
                static_distance = torch.min(static_distance, dim=-1).values
                distance = (active_distance + static_distance) / P
            # refinement with the whole point cloud
            else:
                all_distance = torch.norm(
                    joint.unsqueeze(1) - set_pc[b, t].unsqueeze(0).expand(P-1, -1, -1, 3).reshape(P-1, -1, 3),
                    dim=-1
                    )
                distance = torch.min(all_distance, dim=1).values

            trans_loss.append(distance)
    trans_loss = torch.mean(torch.stack(trans_loss, dim=0), dim=0).sum(dim=0) if trans_loss != [] else 0
    # trans_loss = torch.mean(torch.maximum(torch.abs(trans.flatten()) - 1, torch.zeros_like(trans.flatten())))
    return norm_loss + variance_loss + trans_loss

def segmentation_suppress_loss(label, P):
    min_loss = torch.mean(torch.maximum(
        0.05 - label.flatten(),
        torch.zeros_like(label.flatten())
    ))
    
    max_loss = torch.mean(torch.maximum(
        label.flatten() - (1 - 0.05 * (P-1)),
        torch.zeros_like(label.flatten())
    ))
    
    return min_loss + max_loss