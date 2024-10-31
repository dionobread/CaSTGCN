# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 18:25:40 2024

@author: Gang Zhang
"""

import os
import json
import torch
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset


class CascadeDataset(InMemoryDataset):
    def __init__(self, root, max_nodes=100, sub_size=2, num_feature=4,
                 transform=None, pre_transform=None):
        
        self.max_nodes = max_nodes
        self.num_feature = num_feature
        self.sub_size = sub_size
        
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])
        self.root = root
    
    @property
    def raw_file_names(self):
        
        file_list = os.listdir(self.root + "/raw")
        return file_list
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def process(self):
        '''
        This method processes the raw data by reading JSON files. 
        It creates a list of Data objects containing features, edges, 
        and target values, then it saves the processed data to processed_paths[0].
        Each data object represents a whole cascade graph sample.
        '''
        data_list = []
        
        for filename in tqdm(self.raw_file_names):
            data_path = os.path.join(self.raw_dir, filename)
            
            with open(data_path) as f:
                raw_data = json.load(f)
                num_data = len(raw_data) - self.sub_size
                features = torch.zeros((num_data, self.max_nodes, self.num_feature))
                edges = self.get_edges(raw_data['graph_info'])
                self.in_degree = raw_data['graph_info']['in_degree']
                self.out_degree = raw_data['graph_info']['out_degree']
                target = raw_data['activated_size']
                for i in range(0, num_data):
                    data_key = 'graph_' + str(i)
                    feature = self.get_features(raw_data[data_key])
                    features[i, :, :] = feature
            
            data = Data(x=features, edge_index=edges, y=target)
            data_list.append(data)
        
        self.save(data_list, self.processed_paths[0])
    
    def get_features(self, data):
        
        feature_map = {0.0: 0, 1.0: 1}
        features = torch.zeros((self.max_nodes, self.num_feature))
        node_indices = [self.nodes_map.index(node) for node in data['labels'].keys()]
        feature_indices = [feature_map[label] for label in data['labels'].values()]
        features[node_indices, feature_indices] = 1.0
        for node_index in node_indices:
            features[node_index, self.num_feature-1] = self.in_degree[str(self.nodes_map[node_index])]
            features[node_index, self.num_feature-2] = self.out_degree[str(self.nodes_map[node_index])]
        features = torch.FloatTensor(features)
        
        return features
        
    def get_edges(self, graph_info):
        
        self.nodes_map = [str(nodes_id) for nodes_id in graph_info['nodes']] 
        edges = [[self.nodes_map.index(str(edge[0])), self.nodes_map.index(str(edge[1]))] for edge in graph_info["edges"]]
        
        return torch.t(torch.LongTensor(edges))


if __name__ == "__main__":

    data_name_list = ['dblp', "weibo", "synthetic"]
    
    for data_name in data_name_list:
        data_dir_list = os.listdir("data/" + data_name)
        for data_dir in data_dir_list:
            print("processing", data_name, data_dir, "...")
            root = "data/" + data_name + "/" + data_dir
            if data_name == "synthetic":
                dataset = CascadeDataset(root, sub_size=3)
            else:
                dataset = CascadeDataset(root)


    
    
    
    