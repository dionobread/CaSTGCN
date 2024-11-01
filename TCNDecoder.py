# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 22:58:08 2024

@author: Haobo Liu
"""

import torch.nn as nn
from torch.nn.utils import weight_norm

class TCN_decoder(nn.Module):
    def __init__(self, num_inputs, layer_list, max_len, kernel_size=2, dropout=0.3):
        super().__init__()
        layers = []
    
        num_layers = len(layer_list)
        for i in range(num_layers):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else layer_list[i-1]
            out_channels = layer_list[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                        padding=(kernel_size-1) * dilation_size, dropout=dropout)]
    
        self.network = nn.Sequential(*layers)
        
        # self.mlp = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(layer_list[-1]*max_len, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 1),
        #     nn.ReLU()
        # )
        
    def forward(self, X):
        N, T, H, W = X.shape
        X = X.permute(0, 2, 3, 1) # N H W T
        X = X.view(N, H*W, T)
        X = self.network(X)
        X = X.permute(0, 2, 1)
        
        X = X.view(N, T)
        X = nn.functional.avg_pool1d(X, T)
        # X = self.mlp(X)
        
        return X

 
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
 
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
 
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
 
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
 
    def forward(self, X):
        out = self.net(X)
        res = X if self.downsample is None else self.downsample(X)
        return self.relu(out + res)
 
    
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
 
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
    
