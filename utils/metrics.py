import numpy as np
from cdt.metrics import SHD
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

#----------------------------loss metrics----------------------------
def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe

#----------------------------adj metrics----------------------------
def adj_metrics(pre, ground_truth):
    # print(ground_truth)
    # print(pre)
    # print(type(ground_truth))
    shd = SHD(ground_truth, pre) # 如果你考虑一个反向的边(即，一个矩阵有(i,j)，另一个矩阵有(j,i))，它在SHD中被视为一次改变，而不是两次单独的改变(一次插入和一次删除)。
    # shd = SHD(ground_truth, pre, double_for_anticausal=False)
    precision = precision_score(ground_truth, pre, average='micro')
    recall = recall_score(ground_truth, pre, average='micro')
    F1 = f1_score(ground_truth, pre, average='micro')
    accuracy = accuracy_score(ground_truth.ravel(), pre.ravel())

    # ground_truth_copy = copy.deepcopy(ground_truth)
    # pre_copy = copy.deepcopy(pre_copy)
    # ground_truth_copy = (ground_truth_copy == 1) # change to bool matrix
    # pre_copy = (pre_copy == 1)
    # TP = np.sum(np.sum(pre_copy & ground_truth_copy))
    # FP = np.sum(np.sum(pre_copy & (~ground_truth_copy)))
    # FN = np.sum(np.sum((~pre_copy) & ground_truth_copy))
    # TN = np.sum(np.sum((~pre_copy) & (~ground_truth_copy)))
    # pre_tmp = np.transpose(pre)
    # RA = np.sum(np.sum(pre_tmp & ground_truth))
    # print("TP, FP, FN, TN:", TP, FP, FN, TN)

    # precision = TP / (TP + FP + 1e-9)
    # recall = TP / (TP + FN)
    # F1 = 2 * precision * recall / (precision + recall + 1e-9)
    # accuracy = (TP + TN) / (TP + FP + FN + TN)
    # SHD = FP + FN
    # print(precision, recall, F1, accuracy, SHD)
    return precision, recall, F1, accuracy, shd

import copy
def adj_metrics_FNFP(pre, ground_truth):
    # print(ground_truth)
    # print(pre)
    # print(type(ground_truth))
    shd = SHD(ground_truth, pre) # 如果你考虑一个反向的边(即，一个矩阵有(i,j)，另一个矩阵有(j,i))，它在SHD中被视为一次改变，而不是两次单独的改变(一次插入和一次删除)。
    # shd = SHD(ground_truth, pre, double_for_anticausal=False)
    precision = precision_score(ground_truth, pre, average='micro')
    recall = recall_score(ground_truth, pre, average='micro')
    F1 = f1_score(ground_truth, pre, average='micro')
    accuracy = accuracy_score(ground_truth.ravel(), pre.ravel())

    ground_truth_copy = copy.deepcopy(ground_truth)
    pre_copy = copy.deepcopy(pre)
    ground_truth_copy = (ground_truth_copy == 1) # change to bool matrix
    pre_copy = (pre_copy == 1)
    TP = np.sum(np.sum(pre_copy & ground_truth_copy))
    FP = np.sum(np.sum(pre_copy & (~ground_truth_copy)))
    FN = np.sum(np.sum((~pre_copy) & ground_truth_copy))
    TN = np.sum(np.sum((~pre_copy) & (~ground_truth_copy)))
    # pre_tmp = np.transpose(pre)
    # RA = np.sum(np.sum(pre_tmp & ground_truth))
    # print("TP, FP, FN, TN:", TP, FP, FN, TN)

    # precision = TP / (TP + FP + 1e-9)
    # recall = TP / (TP + FN)
    # F1 = 2 * precision * recall / (precision + recall + 1e-9)
    # accuracy = (TP + TN) / (TP + FP + FN + TN)
    # SHD = FP + FN
    # print(precision, recall, F1, accuracy, SHD)
    return precision, recall, F1, accuracy, shd, FN, FP

def metric(pre, ground_truth):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    add,move,reve = 0,0,0
    for i in range(len(ground_truth)):
        for j in range(len(ground_truth)):
            if pre[i][j] ==ground_truth[i][j] == 1:
                TP = TP + 1
            elif pre[i][j] == ground_truth[i][j]== 0:
                TN = TN + 1
            elif pre[i][j] == 1 and ground_truth[i][j]== 0:
                FP = FP + 1
            elif pre[i][j] == 0 and ground_truth[i][j] == 1:
                FN = FN + 1
    ####shd
            if pre[i][j]!=ground_truth[i][j]:
                if ground_truth[i][j]== 0:
                    if ground_truth[j][i] == pre[i][j]:
                        if i<j:
                            reve = reve+1
                        else:
                            move = move+1
                    else:
                        add = add+1
                else:
                    if ground_truth[i][j]==pre[j][i]:
                        if i <j:
                            reve = reve + 1
                        # else:
                        #     # add = add + 1
                    else:
                        move = move+1

    Fdr = FP/ (TP + FP)
    Tdr = TP/(TP + FN)
    Fpr = FP/(TN + FP)
    Tpr = TP/(TP+FN)  #同召回率
    Accuracy = (TP+TN)/(TP+TN+FP+FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    SHD = add + move +reve

    F = 2*Recall*Precision/(Precision+Recall)

    return Precision, Recall, F, Accuracy, SHD