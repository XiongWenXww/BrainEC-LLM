import numpy as np
import copy
import torch

def softThres(adj, soft_threshold=0.5):
    N = adj.shape[0]
    min_value = 10000000
    # 计算threshold之前就已经将对角线值置为0，故要去除对角线
    for i in range(N):
        for j in range(N):
            if i != j and adj[i, j] < min_value:
                min_value = adj[i, j]
    return min_value + (adj.max() - min_value) * soft_threshold # 用了adj.max()，故需要在这之前吧adj对角线置为0

def change01(adj, threshold):
    np.fill_diagonal(adj, 0)
    adj = np.where(adj >= threshold, 1, 0)
    np.fill_diagonal(adj, 0) # threshold可能为负值，导致可能对角线为1
    return adj

def change01GPU(adj, threshold):
    adj = adj - torch.diag(torch.diag(adj))
    adj = torch.where(adj >= threshold, 1.0, 0.0)
    adj = adj - torch.diag(torch.diag(adj))
    return adj

# row->col
def change01_constraint(adj, threshold, proportion):
    np.fill_diagonal(adj, 0)
    N = adj.shape[0]
    max_num = int(proportion * N)
    processed_adj = np.zeros_like(adj)

    for r in range(N):
        row = adj[r, :]
        
        top_indices = np.argsort(row)[-max_num:]
        top_values = row[top_indices]

        top_values = np.where(top_values > threshold, 1, 0)
        processed_adj[r, top_indices] = top_values
    for c in range(N):
        col = processed_adj[:, c]

        num_ones = np.sum(col == 1)

        if num_ones > max_num:
            col = adj[:, c]
            top_indices = np.argsort(col)[-max_num:]
            top_values = col[top_indices]

            col[:] = 0
            col[top_indices] = 1

            processed_adj[:, c] = col
    return processed_adj

# col->row
# def change01_constraint(adj, threshold, proportion):
#     np.fill_diagonal(adj, 0)
#     N = adj.shape[0]
#     max_num = int(proportion * N)
#     processed_adj = np.zeros_like(adj)

#     for c in range(N):
#         col = adj[:, c]
        
#         top_indices = np.argsort(col)[-max_num:]
#         top_values = col[top_indices]

#         top_values = np.where(top_values > threshold, 1, 0)
#         processed_adj[top_indices, c] = top_values
    
#     for r in range(N):
#         row = processed_adj[r, :]

#         num_ones = np.sum(row == 1)

#         if num_ones > max_num:
#             row = adj[r, :]
#             top_indices = np.argsort(row)[-max_num:]
#             top_values = row[top_indices]

#             row[:] = 0
#             row[top_indices] = 1

#             processed_adj[r, :] = row
#     return processed_adj

# For each row, the maximum max_child_proportion*N values (if these values are greater than the threshold) 
# are set to 1, and other values are set to 0
def change01_constraint_child(adj, threshold, max_child_proportion):
    np.fill_diagonal(adj, 0)
    N = adj.shape[0]
    max_child_num = int(max_child_proportion * N)
    processed_adj = np.zeros_like(adj)

    for r in range(N):
        row = adj[r, :]
        
        top_indices = np.argsort(row)[-max_child_num:]
        top_values = row[top_indices]

        top_values = np.where(top_values > threshold, 1, 0)
        processed_adj[r, top_indices] = top_values
    return processed_adj


# For each column, the maximum max_parent_proportion*N values (if these values are greater than the threshold) 
# are set to 1, and other values are set to 0
def change01_constraint_parent(adj, threshold, max_parent_proportion):
    np.fill_diagonal(adj, 0)
    N = adj.shape[0]
    max_parent_num = int(max_parent_proportion * N)
    processed_adj = np.zeros_like(adj)

    for col in range(N):
        column = adj[:, col]
        
        top_indices = np.argsort(column)[-max_parent_num:]
        top_values = column[top_indices]

        top_values = np.where(top_values > threshold, 1, 0)
        processed_adj[top_indices, col] = top_values
    return processed_adj