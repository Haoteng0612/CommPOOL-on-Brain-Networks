import numpy as np
import pandas as pd
from scipy.io import loadmat
from os.path import join
from utilis.utilis import check_mkdir


data_name = ['network_dim_15.mat', 'network_dim_50.mat', 'network_dim_100.mat', 'network_dim_200.mat', 'network_dim_300.mat']
data_root = '/home/tht/graph/ISBI21/brainnetwork1/'
load_data_dir = join(data_root, 'data')
save_data_dir = join(data_root, 'data/original_data/dim300')         # change here to change the output dir
save_label_dir = join(data_root, 'data/processed_data/node_fea/dim15')
save_node_fea_dir = join(data_root, 'data/processed_data/node_fea')
subj_id_file = '/home/tht/graph/ISBI21/brainnetwork1/data/subjectIDs.txt'
label_file = '/home/tht/graph/ISBI21/brainnetwork1/data/annotations.csv'

def read_txt(txt_file):
    with open(txt_file) as f:
        content = f.readlines()
    num_sub = len(content)
    return content, num_sub

def mat2npy(input_add, output_add, num_sub, sub_list):
    data_file = join(input_add, data_name[4])       # change here to change the input data dir
    data = loadmat(data_file)['matrix']
    if data.shape[2] != num_sub:
        print("The number of the subject is not matched to the number of data")
        raise ValueError
    for i in range(num_sub):
        cur_id = sub_list[i][0:-1]
        save_file_name = join(output_add, str(cur_id)+'.npy')
        cur_data = data[:,:,i]
        print(save_file_name)
        ### save graph adj
        np.save(save_file_name, cur_data, allow_pickle=True, fix_imports=True)
    return num_sub

def process_label(label_file, save_label_dir):
    csv_data = pd.DataFrame(pd.read_csv(label_file))
    num_labels = len(csv_data)
    sub_list, num_sub = read_txt(subj_id_file)
    sub_list = list(map(int, sub_list))
    print(sub_list)
    ### create the labels
    asr = np.zeros(shape=[12])
    bmi = np.zeros(shape=[1])
    dsm = np.zeros(shape=[6])
    handness = np.zeros(shape=[1])
    for i in range(num_labels):
        cur_label_id = csv_data.loc[i][0]
        if cur_label_id in sub_list:
            asr[0], asr[1], asr[2], asr[3], asr[4], asr[5], asr[6], asr[7], asr[8], asr[9], asr[10], asr[11] = \
            csv_data.loc[i][6], csv_data.loc[i][7], csv_data.loc[i][8], csv_data.loc[i][9], csv_data.loc[i][10], \
            csv_data.loc[i][11], csv_data.loc[i][12], csv_data.loc[i][13], csv_data.loc[i][14], csv_data.loc[i][15], \
            csv_data.loc[i][16], csv_data.loc[i][17]

            dsm[0], dsm[1], dsm[2], dsm[3], dsm[4], dsm[5] = \
            csv_data.loc[i][18], csv_data.loc[i][19], csv_data.loc[i][20], csv_data.loc[i][21], csv_data.loc[i][22], csv_data.loc[i][23]

            bmi[0] = csv_data.loc[i][5]

            handness[0] = csv_data.loc[i][3]

            # all_labels = np.concatenate((asr, bmi, dsm, handness), axis=0)
            # if np.isnan(np.sum(all_labels)):
            #     print(all_labels, cur_label_id)

            # save the labels
            # for asr
            # save_asr_file = join(save_label_dir, 'asr', str(cur_label_id) + '.npy')
            # np.save(save_asr_file, asr, allow_pickle=True, fix_imports=True)
            # # for dsm
            # save_dsm_file = join(save_label_dir, 'dsm', str(cur_label_id) + '.npy')
            # np.save(save_dsm_file, dsm, allow_pickle=True, fix_imports=True)
            # # for bmi
            # save_bmi_file = join(save_label_dir, 'bmi', str(cur_label_id) + '.npy')
            # np.save(save_bmi_file, bmi, allow_pickle=True, fix_imports=True)
            # # for handness
            # save_hand_file = join(save_label_dir, 'handness', str(cur_label_id) + '.npy')
            # np.save(save_hand_file, handness, allow_pickle=True, fix_imports=True)

def node_fea_generator(num_nodes, fea_dim, save_dir):
    '''
    generate random node features with Gaussian distributions
    :param num_nodes: N
    :param fea_dim: c
    :return: node feature matrix B * N * c
    '''
    sub_list, num_sub = read_txt(subj_id_file)
    sub_list = list(map(int, sub_list))
    for i in sub_list:
        x = np.random.normal(loc=0, scale=1, size=(num_nodes, fea_dim))
        save_file_dir = join(save_dir, 'dim'+str(num_nodes), 'fea_dim'+str(fea_dim))
        check_mkdir(save_file_dir)
        save_file_name = join(save_file_dir, str(i) + '.npy')
        print(save_file_name)
        np.save(save_file_name, x, allow_pickle=True, fix_imports=True)
    return x








if __name__ == "__main__":
    # sub_list, num_sub = read_txt(subj_id_file)
    # num_sub = mat2npy(load_data_dir, save_data_dir, num_sub, sub_list)
    # process_label(label_file=label_file, save_label_dir=save_label_dir)
    # x = node_fea_generator(num_nodes=300, fea_dim=4, save_dir=save_node_fea_dir)
    pass