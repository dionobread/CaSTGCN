# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 20:51:09 2024

@author: Gang Zhang
"""

import torch
import torch.nn as nn

from STGCNEncoder import st_gcn
from TCNDecoder import TCN_decoder

    
class Model(nn.Module):

    def __init__(self, in_channels, num_nodes, max_len):
        super().__init__()

        # build networks
        # stgcn parameters
        temporal_kernel_size = 5
        kernel_size = temporal_kernel_size
        self.max_len = max_len
        
        # tcn parameters
        in_features = 32 * 100
        layer_list = [128, 128, 64, 1]
        
        self.data_bn =  nn.BatchNorm1d(in_channels*num_nodes)

        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False),
            st_gcn(64, 64, kernel_size, 1),
            st_gcn(64, 32, kernel_size, 1),
            st_gcn(32, 32, kernel_size, 1),
            st_gcn(32, 32, kernel_size, 1)
        ))

        self.tcn_decoder = TCN_decoder(in_features, layer_list, max_len)
        
    def forward(self, data_batch):
        device = data_batch.x.device
        X = self.fill_features(data_batch)
        N, T, H, W = X.shape
        
        # create a list of edges
        edges = []
        for i in range(len(data_batch)):  
            edges.append(data_batch[i].edge_index)

        # data normalization
        X = X.permute(0, 2, 3, 1) # (N, H, W, T)
        X = X.view(N, H * W, T).to(device) # (N, H*W, T)
        X = self.data_bn(X) 
        X = X.view(N, H, W, T)
        X = X.permute(0, 3, 1, 2).contiguous()

        # forward
        # encode
        for stgcn in self.st_gcn_networks:
            X = stgcn(X, edges)
        # decode
        pred = self.tcn_decoder(X)

        return pred
    
    def fill_features(self, data_batch):
        
        N = len(data_batch)
        T = self.max_len
        _, H, W = data_batch[0].x.size()
        
        X = torch.zeros((N, T, H, W))
        # Fill features to the same length as the last cascade graph.
        for i in range(len(data_batch)): 
            feature = data_batch[i].x
            sub = self.max_len - len(feature)
            # fill = feature[-1].repeat(sub, 1, 1)
            fill = torch.zeros((H, W)).repeat(sub, 1, 1).to(data_batch.x.device)
            x = torch.cat((feature, fill), 0)
            X[i, :, :] = x

        return X
    

        