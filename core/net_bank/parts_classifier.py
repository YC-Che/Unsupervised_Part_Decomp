import torch
from torch import nn
from torch.nn import functional as F
from .oflow_point import ResnetPointnet
from .dgcnn import DGCNN

class Parts_classifier(nn.Module):
    def __init__(self, num_t=4, num_p=2):
        super().__init__()
        self.num_p = num_p
        self.num_t = num_t

        self.cls_net=nn.Sequential(*[
            nn.Conv1d(259+7*(self.num_p-1), 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, self.num_p, 1)
        ])

        self.bn = nn.BatchNorm1d(256)

        self.res_point_net = ResnetPointnet(
            c_dim=128,
            dim=3,
            hidden_dim=128,
        )

        #self.dgcnn = DGCNN(emb_dims=256, in_dims=3)
        
    def forward(self, query, c_joint, theta_hat):
        '''
        query: BT,N,3
        c_joint: B,P,6
        theta_hat: B,T,P-1
        '''
        T = theta_hat.shape[1]
        N = query.shape[1]
        P = c_joint.shape[1]
        if c_joint.dim == 3:
            if c_joint.shape[-2] == self.num_p:
                c_j = c_joint[:,:self.num_p-1]#B,P-1,6
            else:
                c_j = c_joint
        else:
            if c_joint.shape[-2] == self.num_p:
                c_j = c_joint[:,:,:self.num_p-1]#B,T,P-1,6
            else:
                c_j = c_joint

        #x = self.dgcnn(query)#BT,N,256
        x_pooled, x = self.res_point_net(query, return_unpooled=True)#BT,N,256
        x = torch.cat((x_pooled.unsqueeze(1).expand(-1,N,-1), x), dim=-1)
        x = self.bn(x.permute(0,2,1)).permute(0,2,1)
        if c_j.dim == 3:
            x = torch.cat([
                query,
                x,
                theta_hat.reshape(-1, self.num_p-1).unsqueeze(1).expand(-1,N,-1), #BT,N,p-1
                c_j.unsqueeze(1).unsqueeze(1).expand(-1,T,N,-1,-1).reshape(-1,N,6*(self.num_p-1)) #BT,N,6(P-1)
            ], dim=-1)#BT,N,256+7(P-1)
        else:
            x = torch.cat([
                query,
                x,
                theta_hat.reshape(-1, self.num_p-1).unsqueeze(1).expand(-1,N,-1), #BT,N,p-1
                c_j.unsqueeze(2).expand(-1,-1,N,-1,-1).reshape(-1,N,6*(self.num_p-1)) #BT,N,6(P-1)
            ], dim=-1)#BT,N,256+7(P-1)
        y = self.cls_net(x.permute(0,2,1))#BT,P,N
        y_softmax = F.softmax(y, dim=1)
        #y_softmax = 0.7 * y_softmax + 0.3/P
        return y_softmax, y
