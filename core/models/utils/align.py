import torch
from torch.nn import functional as F

def binary_split(x):
    B, P, T, N, _ = x.shape
    exclude = [[list(range(i)), list(range(i+1, T))]for i in range(T)]
    exclude = [exclude[i][0] + exclude[i][1] for i in range(T)]  # T,T-1
    others = torch.stack(
        [x.unsqueeze(2)[:, :, :, exclude[i], :, :3] for i in range(T)],
        dim=2)  # B,P,T,T-1,N,3
    others = others.reshape(B*P*T, -1, 3)  # BPT,(T-1)N,3
    query = x.reshape(B*P*T, N, -1)[:, :, :3]  # BPT,N,3
    return query, others

def multi_frame_align(
        x,
        rotation,
        translation,
        cat_frame_label=False
    ):
    '''
    x: B,T,N,3
    rotation: B,P,T,3,3
    translation: B,P,3
    Outout: B,P,T,N,3 or B,P,T,N,(3+T)
    '''
    B, P, T = rotation.shape[:3]
    N = x.shape[2]
    x_multi = x.squeeze(-1).unsqueeze(1).repeat(1, P, 1, 1, 1)  # B,P,T,N,3
    x_multi -= translation.unsqueeze(-2).unsqueeze(-2)
    x_multi = (rotation.unsqueeze(3).expand(-1, -1, -1, N, -1, -1) @ x_multi.unsqueeze(-1)).squeeze(-1)
    x_multi += translation.unsqueeze(-2).unsqueeze(-2)

    if cat_frame_label:
        frame_vec = torch.zeros_like(
            x_multi[:, :, :, :, 0], dtype=torch.long)  # B,P,T,N
        for i in range(T):
            frame_vec[:, :, i, :] = i
        frame_vec = F.one_hot(frame_vec, num_classes=T)
        x_multi = torch.cat(
            [x_multi, frame_vec], dim=-1)  # B,P,T,N,(3+T)

    return x_multi

def rotation_vec_2_matrix(axis, length):
    '''
    axis: B,P,3
    lenght: B,P,T
    mtx: B,P,T,3,3
    '''
    B, P, T = length.shape
    rotation_vec = axis.unsqueeze(2).expand(-1, -1, T, -1) * length.unsqueeze(-1).expand(-1, -1, -1, 3)
    rotation_mtx = ExpSO3(rotation_vec)

    return rotation_mtx


# https://albert.growi.cloud/627ceb255286c3b02691384d
def hat(phi):
    phi_x = phi[..., 0]
    phi_y = phi[..., 1]
    phi_z = phi[..., 2]
    zeros = torch.zeros_like(phi_x)

    phi_hat = torch.stack([
        torch.stack([zeros, -phi_z,  phi_y], dim=-1),
        torch.stack([phi_z,  zeros, -phi_x], dim=-1),
        torch.stack([-phi_y,  phi_x,  zeros], dim=-1)
    ], dim=-2)
    return phi_hat


def ExpSO3(phi, eps=1e-4):
    theta = torch.norm(phi, dim=-1)
    phi_hat = hat(phi)
    E = torch.eye(3, device=phi.device)
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
    return E + coef1 * phi_hat + coef2 * phi_hat @ phi_hat
