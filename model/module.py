from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch.nn import Parameter
import torch
from utilis import utilis as u


class Community_Pool(torch.nn.Module):
    def __init__(self, args, in_channels, ratio, non_linear=torch.tanh):
        super(Community_Pool,self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.non_linear = non_linear
        self.batch_size=args.batch_size
        self.num_node = args.num_node
        self.conv = GCNConv(self.in_channels, self.in_channels)

    def forward(self, istraining, x, edge_index, edge_attr, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        x = self.conv(x, edge_index)
        score, kl_all_batch, indices_ = u.compute_node_score(istraining, x, self.batch_size, self.num_node, batch)
        # print(score.shape, x.shape)
        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linear(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, kl_all_batch, indices_
