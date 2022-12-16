import random

import numpy as np
from os import listdir
from os.path import join

data_dir = '/home/tht/graph/ISBI21/brainnetwork1/data/processed_data/dim15'  # folder of adj, or fea, or label
save_dir = '/home/tht/graph/ISBI21/brainnetwork1/data/processed_data/'

############### define a write file function ###################
def write(data, fname, save_dir=None):
    fname = join(save_dir, fname)
    with open(fname, 'w') as f:
        f.write('\n'.join(data))
################################################################

file_name_list = listdir(data_dir)
print(len(file_name_list))         # file_name_list = [1.npy, 2.npy ...], whole data name
#we have 1003 data point, therefore, can be splitted as 200, 200, 200, 200, 203

# random shuffle
random.shuffle(file_name_list)
# create 5 folder name lists
c0, c1, c2, c3, c4 = file_name_list[0:200], file_name_list[200:400], file_name_list[400:600], file_name_list[600:800], file_name_list[800:1000]


# write in the file
val0, train0 = c0, list(set(file_name_list).difference(set(c0)))   # c0 and non c0 
val1, train1 = c1, list(set(file_name_list).difference(set(c1)))
val2, train2 = c2, list(set(file_name_list).difference(set(c2)))
val3, train3 = c3, list(set(file_name_list).difference(set(c3)))
val4, train4 = c4, list(set(file_name_list).difference(set(c4)))

# 
write(file_name_list, 'all_list.txt', save_dir=save_dir)
write(val0, 'val0_list.txt', save_dir=save_dir)
write(val1, 'val1_list.txt', save_dir=save_dir)
write(val2, 'val2_list.txt', save_dir=save_dir)
write(val3, 'val3_list.txt', save_dir=save_dir)
write(val4, 'val4_list.txt', save_dir=save_dir)
write(train0, 'train0_list.txt', save_dir=save_dir)
write(train1, 'train1_list.txt', save_dir=save_dir)
write(train2, 'train2_list.txt', save_dir=save_dir)
write(train3, 'train3_list.txt', save_dir=save_dir)
write(train4, 'train4_list.txt', save_dir=save_dir)



