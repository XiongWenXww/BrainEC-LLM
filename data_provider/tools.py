import torch
import numpy as np
from scipy.stats import pearsonr

def sliding_window_cutting(data, window_size, overlap):
    step = window_size - overlap
    B, T, N = data.shape

    nums = (T - window_size)//step + 1
    if (T - window_size) % step != 0:
        nums += 1
    new_B = B * nums

    data_sliced = torch.FloatTensor(new_B, window_size, N)
    for b in range(B):
        for num in range(nums - 1):
            data_sliced[b*nums + num, :, :] = data[b, step*num:step*num+window_size, :]
        data_sliced[b * nums + nums - 1, :, :] = data[b, -window_size:, :] # 这使得最后一段的数据和其它段的数据长度相同

    return data_sliced

def calculate_pearson_correlation(data):
    """
    Calculate the Pearson correlation coefficient matrix for brain regions.

    Parameters:
    data (numpy.ndarray): Input data with shape [M, T, N]
        M: Number of samples
        T: Number of time steps
        N: Number of brain regions

    Returns:
    numpy.ndarray: Pearson correlation coefficient matrix with shape [N, N]
    """
    M, T, N = data.shape
    correlation_matrix = np.zeros((N, N))
    
    for m in range(M):
        sample = data[m]
        corr_matrix = np.corrcoef(sample, rowvar=False)
        correlation_matrix += corr_matrix
    
    correlation_matrix /= M
    np.fill_diagonal(correlation_matrix, 0)
    
    return correlation_matrix