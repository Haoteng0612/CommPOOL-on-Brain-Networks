'''You should change this file w.r.t. your dataset'''
import argparse
from typing import Any

from dataset.dataset import TrainDataset
from datetime import datetime
import os
from os.path import join
from utilis.utilis import check_mkdir, grid_2_COO


# model import
from model.networks import Net
import torch.optim as optim
import torch.optim
# from torch_geometric.data import DataLoader as DataLoaderT
from torch_geometric import utils

import numpy as np
import random
from tensorboardX import SummaryWriter
import torch
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader
import time

exp_name = 'HCP'
datadir = ''     # where data save
labeldir = ''       # where label save
feadir = ''
save_path = ''
runs_path = ''

cudnn.benchmark = True
check_mkdir(save_path)

def write(data, fname, save_dir=None):
    fname = join(save_dir, fname)
    with open(fname, 'w') as f:
        f.write(data + '\n')

def main():
    parser = argparse.ArgumentParser(description="HCP brain network classification")
    parser.add_argument('--graphuse', type=str, default='dim50', choices=('dim15', 'dim50', 'dim100', 'dim200', 'dim300')) # change here to change adj input
    parser.add_argument('--num_node', type=int, default='50', choices=(15, 50, 100, 200, 300))  # change here to change adj input
    parser.add_argument('--featureuse', type=str, default='fea_dim4', choices=('fea_dim4','fea_dim8'))    # change here to change feature input
    parser.add_argument('--labeluse', type=str, default='asr', choices=('asr', 'bmi', 'dsm', 'handness'))  # change here to change task
    parser.add_argument('--batch_size', type=int, default=256)              # change batch size
    parser.add_argument('--gpu', type=str, default=('0'))                   # change gpu
    parser.add_argument('--cuda', action='store_true', default=True)        # if use cuda
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--nEpochs', type=int, default=5000)                # change number of training epochs
    parser.add_argument('--change_opt', action='store_true', default=False)
    parser.add_argument('--learning_rate', type=float, default=1e-6)        # change learning rate
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--initial_type', type=str, default='none')
    parser.add_argument('--eps', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='adam', choices=('sgd', 'adam'))
    parser.add_argument('--resume', default='')
    parser.add_argument('--train_list', type=str, default='train0_list',
                        choices=('train0_list', 'train1_list', 'train2_list',
                                 'train3_list', 'train4_list'))
    parser.add_argument('--valid_list', type=str, default='val0_list',
                        choices=('val0_list', 'val1_list', 'val2_list',
                                 'val3_list', 'val4_list'))
    parser.add_argument('--datadir', type=str, default=datadir)
    parser.add_argument('--labeldir', type=str, default=labeldir)
    parser.add_argument('--feadir', type=str, default=feadir)

    parser.add_argument('--num_features', type=int, default=4, choices=(4, 8))
    parser.add_argument('--num_classes', type=int, default=1, choices=(1, 12, 6))      # number of the prediction that should match the label size
    parser.add_argument('--nhid', type=int, default=64, choices=(64, 128))
    parser.add_argument('--pooling_ratio', type=float, default=0.5)
    parser.add_argument('--dropout_ratio', type=float, default=0.5)
    parser.add_argument('--loss_type', type=str, default='regression', choices=('regression', 'classification'))


    args = parser.parse_args()

    #*****************************************  handle cuda, gpu, seed for all *****************************************#
    args.cuda = args.cuda and torch.cuda.is_available()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(args.gpu)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    # ***************************************************************************************************************** #

    # ********************************************  set up the model   ************************************************ #
    model = Net(args=args)
    print("Using SAG pooling Model")
    # ***************************************************************************************************************** #

    # ********************************************  set up the optimizer ********************************************** #
    if args.opt == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate,
                              momentum=0.99, weight_decay=args.weight_decay)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.learning_rate, eps=args.eps, weight_decay=args.weight_decay)
    # ***************************************************************************************************************** #
    # ********************************************  cuda parallel, hang model on cuda ********************************* #
    if args.cuda:
        model = model.cuda()     # hang the model on the GPU
        model = torch.nn.DataParallel(model, device_ids=[0])
    # ***************************************************************************************************************** #

    # ******************************************** load data ********************************************************** #
    print('==================> Loading training datasets on: {}'.format(args.train_list))
    train_set = TrainDataset(train_list=args.train_list, datadir=args.datadir, labeldir=args.labeldir, feadir=args.feadir, args=args, transforms=None, is_train=True)
    val_set = TrainDataset(train_list=args.valid_list, datadir=args.datadir, labeldir=args.labeldir, feadir=args.feadir, args=args, transforms=None, is_train=False)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                              pin_memory=False, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_set, batch_size=1, num_workers=args.num_workers, pin_memory=False, shuffle=False, drop_last=True)
    # for batch_idx, (adj, fea, y, batch_index) in enumerate(val_loader):
    #     print(adj, fea.shape, y, batch_index)


    torch.set_grad_enabled(True)
    write_file_name = join(save_path, 'validation_results.txt')
    f = open(write_file_name, 'w')
    f.write('Current num_node is {}, feature dim is {}, label is {}, nEpoch is {}, lr is {}, nhid is {}, Pool_r is {}, drop_r is {} \n' \
            .format(args.num_node, args.num_features, args.labeluse, args.nEpochs, args.learning_rate, args.nhid, args.pooling_ratio, args.dropout_ratio))
    for epoch in range(args.nEpochs):
        adjust_learning_rate(optimizer, epoch, max_epoch=args.nEpochs, init_lr=args.learning_rate, warmup_epoch=5)
        print("current learning is {}".format(optimizer.param_groups[0]['lr']))
        cluster_indices = train(args=args, epoch=epoch, model=model, trainloader=train_loader, optimizer=optimizer, istraining=True)
        if args.valid_list is not None and epoch % 10 == 0 and epoch != 0:
            ############## save the cluster information #################
            write_cluster_file_name = join(save_path, 'cluster_info_epoch_{}'.format(epoch))
            np.save(write_cluster_file_name, cluster_indices)
            with torch.no_grad():
                loss_count = valid(args=args, epoch=epoch, model=model, val_loader=val_loader, optimizer=optimizer, istraining=False)
                ############## write in the validation results in txt ###############
                write_in = 'The current epoch is {} and the val loss in this epoch is {} \n'.format(epoch, loss_count)
                f.write(write_in)
                torch.cuda.empty_cache()
    f.close()





    # ******************************************* define  training **************************************************** #
def train(args, epoch, model, trainloader, optimizer, istraining=True):
    istraining = istraining
    model.train()
    print("----------> current epoch is {}".format(epoch))
    mean_loss = 0
    cluster_indices = {}
    for batch_idx, (adj, fea, y, batch_index) in enumerate(trainloader):
        A, x, label, batch_index = adj, fea.type(torch.FloatTensor), y.type(torch.FloatTensor), batch_index[0]
        A, x, label, batch_index = A.cuda(), x.cuda(), label.cuda(), batch_index.cuda()
        edge_index, edge_weight, x = grid_2_COO(x, A)    # A is in shape of N * N
        out, loss, kl_all, indices_ = model(istraining, fea=x, adj_index=edge_index, adj_weight=edge_weight, batch_index=batch_index, label=label)
        cluster_indices[batch_index] = indices_
        whole_loss = loss + kl_all
        mean_loss += loss
        # print("Training loss:{}".format(loss.item()))
        whole_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print('Training loss in this batch is {}'.format(mean_loss/ (int(800 / args.batch_size)+1)))
    return cluster_indices

    # ***************************************************************************************************************** #
    # ******************************************* define  validation ************************************************** #
def valid(args, epoch, model, val_loader, optimizer, istraining=False):
    istraining = istraining
    model.eval()
    mse_count = 0.
    loss_count = 0.
    for batch_idx, (adj, fea, y, batch_index) in enumerate(val_loader):
        A, x, label, batch_index = adj, fea.type(torch.FloatTensor), y.type(torch.FloatTensor), batch_index[0]
        A, x, label, batch_index = A.cuda(), x.cuda(), label.cuda(), batch_index.cuda()
        edge_index, edge_weight, x = grid_2_COO(x, A)
        out, loss, _, _ = model(istraining, fea=x, adj_index=edge_index, adj_weight=edge_weight, batch_index=batch_index, label=label)
        if torch.isnan(loss):
            print(out, label, loss, batch_idx)
            raise ValueError
        loss_count += loss.item()
    loss_count = loss_count / 200
    print("**************** validation loss is {} ****************".format(loss_count))
    return loss_count

def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, warmup_epoch, power=0.9):
    if epoch < warmup_epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = round(init_lr * min(1.0, epoch / warmup_epoch), 8)
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = round(init_lr * np.power(1 - (epoch - warmup_epoch) / max_epoch, power), 8)




if __name__ == '__main__':
    main()







