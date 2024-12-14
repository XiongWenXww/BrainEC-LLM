import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from data_provider.tools import sliding_window_cutting, calculate_pearson_correlation
import warnings

warnings.filterwarnings('ignore')

def data_path_gd(args):
    index = str(args.index)
    if args.data == 'Smith':
        ground_truth = np.loadtxt(
            args.path + 'ground_truth' + index + '.txt', delimiter='\t')
        path = args.path + 'sim' + index + '/'
        skiprows = 0
    elif args.data == 'Sanchez':
        if index == '8' or index == '9':
            path = args.path + 'Network' + index + '_amp_amp/data_fslfilter/'
        else:
            path = args.path + 'Network' + index + '_amp/data_fslfilter/'
        ground_truth = np.loadtxt(args.path + 'ground_truth' + index + '.txt', delimiter='\t')
        skiprows = 1
    elif args.data == 'Real':
        path = args.path + 'individual_' + args.pos + '_mtl_reduced/'
        ground_truth = np.loadtxt(args.path + args.pos  + '_ground_truth.txt', delimiter='\t')
        skiprows = 1
    else:
        print('There is no such dataset')
    
    return path, ground_truth, skiprows

def load_Smith_Sanchez_real(args):
    path, ground_truth, skiprows = data_path_gd(args)
    
    all_path = os.listdir(path)
    subjects = len(all_path)
    data = np.empty((subjects, 0, 0))
    i = 0
    for sub_path in all_path:
        position = path + sub_path
        data_tmp = np.loadtxt(position, skiprows=skiprows, delimiter='\t')
        if i == 0:
            data = np.expand_dims(data_tmp, axis=0)
        else:
            data = np.concatenate((data, np.expand_dims(data_tmp, axis=0)), axis=0)
        i += 1
    return data, ground_truth

def load_generate(args):
    path = args.path + 'seed42_node' + str(args.index) + '_n_samples10000/'
    ground_truth = np.load(path + 'ground_truth.npy')
    all_path = os.listdir(path)
    subjects = len(all_path) - 1
    print('subjects:', subjects)
    data = np.empty((subjects, 0, 0))
    for subject in range(1, subjects+1):
        position = path + 'data' + str(subject) + '.npy'
        data_tmp = np.load(position)
        if subject == 1:
            data = np.expand_dims(data_tmp, axis=0)
        else:
            data = np.concatenate((data, np.expand_dims(data_tmp, axis=0)), axis=0)
    return data, ground_truth

def prepare_dataloaders(args):
    if args.data == 'CausalDiscovery_linear' or args.data == 'CausalDiscovery_quadratic':
        data, ground_truth = load_generate(args)
    elif args.data == 'Smith' or args.data == 'Sanchez' or args.data == 'Real':
        data, ground_truth = load_Smith_Sanchez_real(args)
    
    print("data:", data.shape) # [S, T, N]
    args.adj = calculate_pearson_correlation(data)
    print('adj:', args.adj)

    if args.is_norm:
        min_T = np.min(data, axis=1, keepdims=True) # [B, 1, N]
        max_T = np.max(data, axis=1, keepdims=True)

        range_T = max_T - min_T
        range_T[range_T == 0] = 1 # 防止除以零的情况
        data = (data - min_T) / range_T
    
    data = torch.FloatTensor(data).to(args.device) # data:[S, T, N], S:number of subjects
    args.nodes_num = data.shape[2] # N

    # sliding window method
    if args.sliding_window == 1:
        args.ts_len = args.window_size
        data_sliced = sliding_window_cutting(data=data, window_size=args.window_size, overlap=args.overlap)
        data_sliced = torch.FloatTensor(data_sliced).to(args.device)
        label = data_sliced
        
        print("data_sliced:", data_sliced.shape)
        dataset = TensorDataset(data_sliced, label)
    else:
        args.ts_len = data.shape[1]  # T
        label = data
        dataset = TensorDataset(data, label)
    args.VAGE_in_dim = args.ts_len # T

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle) # 设置seed可以保证每个epoch得到的数据顺序一样
    return data_loader, ground_truth