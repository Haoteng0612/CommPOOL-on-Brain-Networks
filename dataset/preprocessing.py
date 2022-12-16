import numpy as np
from os import listdir
from os.path import join

data_dir = '/home/tht/graph/ISBI21/brainnetwork1/data/original_data/dim300'
save_dir = '/home/tht/graph/ISBI21/brainnetwork1/data/processed_data'
file_names = listdir(data_dir)
print(file_names)

###### invert z-transform #####
for files in file_names:
    cur_file = join(data_dir, files)
    print(cur_file)
    adj = np.load(cur_file)
    adj = np.tanh(adj) + 1
    np.save(join(save_dir, 'dim300', files), adj, allow_pickle=True, fix_imports=True)