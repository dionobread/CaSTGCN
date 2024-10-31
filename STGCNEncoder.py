# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:21:48 2024

@author: Gang Zhang
"""
import torch
from torch import nn


class st_gcn(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1,
                 dropout=0.3, residual=True):
        super().__init__()

        padding = ((kernel_size - 1) // 2, 0)
        self.out_channels = out_channels
        
        self.att1 = nn.Conv2d(in_channels, 1, 1)
        self.att2 = nn.Conv2d(in_channels, 1, 1)

        self.leaky_relu = nn.LeakyReLU()
        
        self.gcn = GCNConv(in_channels, out_channels)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(kernel_size, 1),
                stride=(stride, 1),
                padding=padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, X, edges):
        _, T, _, _ = X.shape
        self.device = X.device

        adj = self.edge_index_to_adj(X, edges)
        adj = adj.unsqueeze(1).repeat(1, T, 1, 1)
        
        # compute the attention value
        att1 = self.att1(X.permute(0, 3, 2, 1)).permute(0, 2, 1, 3)
        att2 = self.att2(X.permute(0, 3, 2, 1))

        att = nn.functional.softmax(self.leaky_relu(att1 + att2), dim=1).permute(0, 3, 1, 2)
        
        res = self.residual(X.permute(0, 3, 1, 2))
        
        X = self.gcn(X, adj, att)
        
        X = self.tcn(X.permute(0, 3, 1, 2)) + res
        
        return self.relu(X.permute(0, 2, 3, 1))
    
    def edge_index_to_adj(self, X, edges):
        
        _, _, H, _ = X.shape
        adj = torch.tensor([]).to(self.device)
        for i, edge_index in enumerate(edges):
            v = torch.ones(edge_index.size(1)).to(self.device)
            coo = torch.sparse_coo_tensor(edge_index, v, (H, H))
            adj = torch.cat([adj, coo.to_dense().unsqueeze(0)], 0)
            
        return adj
    
    
class GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.lin = nn.Linear(in_channels, out_channels)
    
    def forward(self, X, adj, edge_weight):
        self.device = X.device
        adj = torch.matmul(adj, edge_weight)
        
        adj = self.add_self_loops(adj)
        
        support = self.lin(X)
        output = torch.einsum('nthh,ntho->ntho', adj, support)
        
        return output
    
    def add_self_loops(self, adj):
        N, T, H, _ = adj.shape
        
        identity = torch.eye(H, H).repeat(N, T, 1, 1).to(self.device)
        
        return adj + identity
        

        
        
    