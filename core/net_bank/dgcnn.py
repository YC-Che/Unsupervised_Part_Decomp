#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""


import os
import sys
import copy
#import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, only_xyz=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if only_xyz:
            idx = knn(x[:,:3,:], k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
    #device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature

class DGCNN(nn.Module):
    def __init__(self, k=20, emb_dims=128, dropout=0., in_dims=3):
        super(DGCNN, self).__init__()
        self.emb_dims = emb_dims
        self.dropout = dropout
        self.k = k
        

        self.conv1 = nn.Sequential(nn.Conv2d(2*in_dims, 64, kernel_size=1, bias=False),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.mlp = nn.Sequential(nn.Conv1d(self.emb_dims, 32, 1),
                                 nn.LeakyReLU(),
                                 nn.Conv1d(32, 32, 1),
                                 nn.LeakyReLU(),
                                 nn.Conv1d(32, 1, 1))

    def forward(self, x):
        batch_size = x.size(0)
        x = x.permute(0,2,1)
        x = get_graph_feature(x, k=self.k, only_xyz=True)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)#BP,emb_dim,TN
        #x = self.mlp(x)#BP,1,TN

        return x.permute(0,2,1)

def main():
    input = torch.randn((32, 500, 7)).cuda()
    model = DGCNN(emb_dims=256).cuda()
    with torch.no_grad():
        output = model(input)
        print(output.shape)
if __name__ == "__main__":
    main()