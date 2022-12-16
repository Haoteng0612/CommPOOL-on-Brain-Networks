import torch
from torch.utils.data import Dataset
import numpy as np
from os.path import join
import os

class TrainDataset(Dataset):
    def __init__(self, train_list, datadir=None, labeldir=None, feadir=None, args=None, transforms=None, is_train=True):
        # where your data is?
        graph_dir = join(datadir, args.graphuse)  # where your adj is?
        label_dir = join(labeldir, args.labeluse) # where your label is?
        fea_dir = join(feadir,args.graphuse,args.featureuse) # where your feature is?

        # read the data file name from the txt one by one and save the data name in the name list
        names = []
        with open(join(datadir, train_list + '.txt')) as f:
            for line in f:
                line = line.strip()
                name = line
                names.append(name)       

        self.names = names
        self.label_dir = label_dir
        self.graph_dir = graph_dir
        self.fea_dir = fea_dir
        self.args = args
        self.transform = transforms
        self.is_train = is_train

    def __getitem__(self, index):
        # load the data based on (1) name list, (2) current index
        y = np.load(join(self.label_dir, self.names[index]))
        # if np.isnan(y):
        #     y = np.array([3.14])
        adj = np.load(join(self.graph_dir, self.names[index]))
        fea = np.load(join(self.fea_dir, self.names[index]))

        # generate the batch_id
        if self.is_train:
            num_batch, num_node = self.args.batch_size, fea.shape[0]
        else:
            num_batch, num_node = 1, fea.shape[0]

        batch_index = np.arange(num_batch)     # [1, 2, 3, ..., num_batch]
        batch_index = np.repeat(batch_index, num_node) # [1, 1, ..., 1, 2, 2, ..., 2, ......, 128, 128, ...128]

        # convert the numpy array to the pytorch array
        adj, fea, y, batch_index = torch.from_numpy(adj), torch.from_numpy(fea), torch.from_numpy(y), torch.from_numpy(batch_index)

        return adj, fea, y, batch_index

    def __len__(self):
        return len(self.names)
