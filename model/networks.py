import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
# from torch_geometric.nn.pool import SAGPooling as SAG_P
from model.module import Community_Pool as CmP

class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.loss_type = args.loss_type
        self.conv_p = GCNConv(self.nhid, 32)

        self.conv1 = GCNConv(self.num_features, self.nhid)
        # args, in_channels, ratio, Conv = GCNConv, non_linear = torch.tanh
        self.pool1 = CmP(args=self.args, in_channels=self.nhid, ratio=self.pooling_ratio, non_linear=torch.tanh)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.pool2 = CmP(args=self.args, in_channels=self.nhid, ratio=self.pooling_ratio, non_linear=torch.tanh)
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.pool3 = CmP(args=self.args, in_channels=self.nhid, ratio=self.pooling_ratio, non_linear=torch.tanh)

        self.lin1 = torch.nn.Linear(self.nhid*2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.lin3 = torch.nn.Linear(self.nhid//2, self.num_classes)

        self.loss1 = nn.L1Loss()
        self.loss2 = nn.MSELoss()

    def forward(self, istraining, fea, adj_index, adj_weight, batch_index, label):
        label = label[:,0].unsqueeze(1)
        x, edge_index, edge_weight, batch = fea, adj_index, adj_weight, batch_index

        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x, edge_index, edge_weight, batch, _, kl_all_batch_poo1, indices_ = self.pool1(istraining, x, edge_index, edge_weight, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        #input --> graph convolution1 --> graph pooling1 --> summary[global_max_pool(x), global_mean_pool(x)]=x1 

        # print(x.shape, edge_index.shape, edge_weight.shape)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x, edge_index, edge_weight, batch, _, kl_all_batch_poo2, _ = self.pool2(istraining, x, edge_index, edge_weight, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_weight))
        x, edge_index, edge_weight, batch, _, kl_all_batch_poo3, _ = self.pool3(istraining, x, edge_index, edge_weight, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        kl_all = kl_all_batch_poo1 + kl_all_batch_poo2 + kl_all_batch_poo3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        out = F.log_softmax(x, dim=-1)
        # print(label)
        # print(x.type(), label.type())

        # compute MAE loss for regression
        if self.loss_type == 'classification':
            loss = F.nll_loss(out, label)
            return out, loss

        if self.loss_type == 'regression':
            loss = self.loss1(x, label)
            return x, loss, kl_all, indices_



