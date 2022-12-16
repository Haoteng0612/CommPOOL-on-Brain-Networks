import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import torch

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def compute_node_score(istraining, x=None, batch_size=None, num_node=None, batch=None):
    '''
    given x, output the score for each node with the community information
    :param x: BN * c
    :return:
    '''
    # if x.shape[0] != batch_size * num_node:
    #     print('BN is not equal to batch_size * num_node')
    #     raise ValueError
    BN, fea_dim = x.shape[0], x.shape[1]
    score_cat = torch.zeros([1, 1], device=x.device)
    batch_ = batch.unsqueeze(1)
    batch_x = torch.cat((batch_, x), 1)
    kl_all_batch = 0
    node_center_sum = []
    for idx in range(batch_size):
        x_mask = batch_x[batch_x[:,0]==idx]      # pick up the node feature of batch idx
        x_mask = x_mask[:,1:]         # N * c
        x_mask_sum = torch.sum(x_mask, dim=1).unsqueeze(1)    # N * 1
        x_mask_t = x_mask.t()        # c * N
        supp1 = torch.mm(x_mask_t, x_mask_sum)  # c * 1
        score = torch.mm(x_mask, supp1)  # N * 1

        if istraining:
            kl_loss_graph, indices_ = compute_kl_loss(score, x_mask, k_ratio=0.5)
            kl_all_batch += kl_loss_graph
            node_center_sum.append(indices_)

        score_cat = torch.cat((score_cat, score))
    score_cat = score_cat[1:,]    # BN * 1
    score_cat = F.normalize(input=score_cat, p=2.0, dim=0).squeeze()    # BN
    score_cat = 1 - score_cat

    # # x = x.view(batch_size, num_node, fea_dim)      # B * N * c
    # x_sum = torch.sum(x, dim=2).unsqueeze(2)       # B * N * 1
    # x_t = x.permute(0, 2, 1)   # B * c * N
    # supp1 = torch.bmm(x_t, x_sum)    # B * c * 1
    # score = torch.bmm(x, supp1)      # B * N * 1
    # score = score.view(batch_size*num_node, 1)     # BN * 1
    # score = F.normalize(input=score, p=2.0, dim=0).squeeze()
    return score_cat, kl_all_batch, node_center_sum


def compute_kl_loss(score, x_mask, k_ratio=0.5):
    '''

    :param score: N * 1
    :param x: N * c
    :return: kl loss for each graph
    '''
    # KL_graph = nn.KLDivLoss(size_average=True, reduce=True)
    # print(x_mask.shape)
    L1_graph = nn.L1Loss()
    num_node = x_mask.shape[0]
    sim_matrix = torch.mm(x_mask, x_mask.t())   # N * N
    # print('sim matrix {}'.format(sim_matrix.shape))
    # compute the top K score
    K = int(k_ratio * x_mask.shape[0])
    values, indices = torch.topk(score.squeeze(), K)
    sim_matrix_select = torch.index_select(sim_matrix, 1, indices)        # N * K
    # print(sim_matrix_select.shape)
    values_, indices_ = torch.max(sim_matrix_select, 1)
    fea_j = torch.zeros([x_mask.shape[0], x_mask.shape[1]], device=x_mask.device, requires_grad=False)
    for node_j in range(num_node):
        center_indx = indices_[node_j]
        fea_j[node_j,:] = x_mask[center_indx,:]
    # compute KL loss for each graph
    kl_loss_graph = L1_graph(x_mask, fea_j)
    # print("kl_loss for this graph is {}".format(kl_loss_graph))
    return kl_loss_graph, indices_



def grid_2_COO(x, A):
    """
    Create batch concatenated node feature x and COO edge_index
    Args:
        x (Tensor): original node features. A tensor of shape (B, N, C).
        att_soft (Tensor): original fully connected weighted adjacent matrix. A tensor of shape (B, N, N).
        candid_mask (Tensor): Uncertain pixels index. A tensor of shape (B, N).
    Returns:
        par_indices, par_values (Tensor): COO edge_index and the connectivity values. tensors of
        shape (2, B*N) and (1, B*N).
    """
    # adj_sparse = torch.exp_(torch.neg(adj_sparse))
    batch_x, batch_adj = x.shape[0], A.shape[0]
    if batch_x != batch_adj:
        print("batch size is not matched !!")
        raise ValueError
    else:
        b = int(batch_x)  # b

    num_node, fea_dim = int(x.shape[1]), int(x.shape[2])  # hw

    #### process adj
    # set diag to 0
    diag = torch.eye(num_node, num_node).to(device=A.device)  # b, hw, hw

    adj_sparse = A - A * diag
    del diag

    start_node = torch.arange(b, dtype=torch.long, device=adj_sparse.device) * num_node

    start_node = start_node.unsqueeze(-1).unsqueeze(-1) * torch.ones(adj_sparse.shape, dtype=torch.long,
                                                                     device=adj_sparse.device)  # b, n, n
    start_node = start_node[adj_sparse != 0]  # m
    start_node = start_node.repeat(2, 1)  # 2, m

    # obtain edge index
    edge_index = torch.nonzero(adj_sparse, as_tuple=False)  # m, 3
    edge_index = edge_index.permute(1, 0).contiguous()  # 3, m

    # obtain edge weight
    edge_weight = adj_sparse[adj_sparse != 0]  # m

    if edge_weight.shape[0] != edge_index.shape[1]:
        raise ValueError
    edge_weight = edge_weight.view(1, -1)  # 1, m
    edge_weight = edge_weight.squeeze()

    # obtain edge index weight
    edge_index = edge_index[1:] + start_node

    # conver H: B*N*C to H: BN * C
    x = x.view(batch_x*num_node, fea_dim)



    return edge_index.long(), edge_weight.float(), x