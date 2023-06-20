from .oflow_point import ResnetPointnet
import torch
from torch import nn
from torch.nn import functional as F
from .dgcnn import DGCNN

class Joint_estimator(nn.Module):
    def __init__(
        self,
        atc_num,
        c_dim=256,
        ci_dim=256,
        hidden_dim=256,
    ) -> None:
        super().__init__()
        self.atc_num = atc_num
        self.backbone_pointnet = ResnetPointnet(
            dim=3,
            c_dim=c_dim,
            hidden_dim=hidden_dim,
        )
        self.set_mlp_layers = nn.ModuleList(
            [nn.Linear(c_dim * 2, c_dim), nn.Linear(c_dim * 2, c_dim), nn.Linear(c_dim * 2, c_dim)]
        )

        self.theta_fc = nn.Sequential(
            nn.Linear(c_dim, c_dim),
            nn.ReLU(), 
            nn.Linear(c_dim, c_dim),
            nn.ReLU(),
            nn.Linear(c_dim, atc_num))
        
        self.axis_net=nn.Sequential(*[
            nn.Conv1d(515, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 1)
        ])

        self.axis_fc = nn.Sequential(
            nn.Linear(256, 128), 
            nn.ReLU(), 
            nn.Linear(128, 6 * (atc_num+1)), 
            nn.Linear(6 * (atc_num+1), 6 * (atc_num+1)))

        self.confidence_fc = nn.Sequential(
            nn.Linear(c_dim, c_dim), 
            nn.ReLU(), 
            nn.Linear(c_dim, c_dim), 
            nn.ReLU(), 
            nn.Linear(c_dim, 1))

        self.bn = nn.BatchNorm1d(256)

        #self.dgcnn_encoder = DGCNN(k=20, emb_dims=256, dropout=0., in_dims=3)
    
    def forward(self, pc_set, category):
        B,T,N,_ = pc_set.shape

        x = pc_set.reshape(B*T, N, -1)#BT,N,3
        #y = self.dgcnn_encoder(x)#BT,N,256
        _, y = self.backbone_pointnet(x, return_unpooled=True)#BT,N,256
        y = self.bn(y.permute(0,2,1)).permute(0,2,1)
        f_global = torch.max(y, dim=1).values#BT, 256

        theta_hat = self.theta_fc(f_global).reshape(B,T,-1)
        confidence = self.confidence_fc(f_global).reshape(B,T,-1)#B,T,1
        confidence = F.softmax(confidence, dim=1).squeeze(-1)#B,T
        axis = self.axis_net(torch.cat(
            (x, y, f_global.unsqueeze(1).expand(-1,N,-1)), dim=-1).permute(0,2,1)
            )
        axis = torch.max(axis, dim=-1).values
        axis = self.axis_fc(axis).reshape(B,T,-1,6)
        if category in ['scissors']:
            axis[:, :, 0, :] = axis[:, :, 1, :].clone()
        #if category in ['eyeglasses']:
        #    axis[:, :, 0, :3] = axis[:, :, 1, :3].clone()
        axis_mean = torch.mean(axis, dim=1)

        return axis_mean, theta_hat, confidence, axis
