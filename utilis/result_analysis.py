import torch
import numpy as np
import os
from os.path import join

from torch_geometric.utils import to_dense_adj

result_dir = '/home/tht/graph/ISBI21/brainnetwork1/model_save/HCP'
date_dir = 'Sep04_20-47-07'

def generate_cluster_matrix(cluster_index):
    num_node = cluster_index.shape[0]
    node_index = torch.arange(num_node)
    node_index, cluster_index = node_index.unsqueeze(0).cpu(), cluster_index.unsqueeze(0).cpu()
    edge_index = torch.cat((node_index, cluster_index), 0)
    print(cluster_index)
    adj = to_dense_adj(edge_index).squeeze()
    print(adj.sum())
    return adj


file_names = join(result_dir, date_dir, 'cluster_info_epoch_200.npy')
print(file_names)
data = np.load(file_names, allow_pickle=True).item().values()
data = list(data)
adj = torch.zeros(size=[50, 50])
for len_data in range(len(data)):
    cur_data = data[len_data]
    for len_cur_data in range(len(cur_data)):
        cluter_index_matrix = cur_data[len_cur_data]
        adj1 = generate_cluster_matrix(cluter_index_matrix)
        adj2 = adj1.t()
        adj = adj + adj1 + adj2
print(adj.shape)

values, target = torch.max(adj,0)
print(target)




