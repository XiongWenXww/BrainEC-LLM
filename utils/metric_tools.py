import numpy as np
import copy
from scipy.stats import pearsonr
import scipy.stats
import matplotlib.pyplot as plt
import fnmatch
import os
import scipy.io as sio
import matplotlib.ticker as ticker
def change01(adj, threshold):
    alpha = copy.deepcopy(adj)
    N = alpha.shape[0]
    # 最后令对角线为0,否则可能th小于0会使得后面对角线又为0了
    alpha = np.where(alpha >= threshold, 1, 0)
    for i in range(N):
        alpha[i, i] = 0

    # adj[np.arange(N), np.arange(N)] = 0
    # 这些代码试图使用布尔类型的索引来选择满足条件的元素，
    # 并将它们分别设置为1和0。但是在NumPy中，布尔类型的索引是不适用于直接赋值的。
    # index = (alpha >= threshold)
    # alpha[index] = 1
    # index = (alpha < threshold)
    # alpha[index] = 0
    return alpha

def cal_metrics(pre, ground_truth):
    ground_truth = (ground_truth == 1) # change to bool matrix
    pre = (pre == 1)
    TP = np.sum(np.sum(pre & ground_truth))
    FP = np.sum(np.sum(pre & (~ground_truth)))
    # print(pre & (~ground_truth))
    FN = np.sum(np.sum((~pre) & ground_truth))
    TN = np.sum(np.sum((~pre) & (~ground_truth)))
    # pre_tmp = np.transpose(pre)
    # RA = np.sum(np.sum(pre_tmp & ground_truth))
    # print("TP, FP, FN, TN:", TP, FP, FN, TN)
    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    SHD = FP + FN
    # print(f'FN:{FN}, FP:{FP}')
    # print(precision, recall, F1, accuracy, SHD)
    return precision, recall, F1, accuracy, SHD, FN, FP

def softThres(adj, soft_threshold):
    N = adj.shape[0]
    min_value = 10000000
    # 计算threshold之前就已经将对角线值置为0，故要去除对角线
    for i in range(N):
        for j in range(N):
            if i !=j and adj[i, j] < min_value:
                min_value = adj[i, j]
    return min_value + (adj.max() - min_value) * soft_threshold # 用了adj.max()，故需要在这之前吧adj对角线置为0

# 得到方法在给定的sanch数据集上的所有评价指标
def metrics(way, indexs, method, th):
    metrics = []
    for index in indexs:
        gd = sanch_groundtruth(index)
        path = './' + way + '/' + method + '/sanch/sim' + str(index) + '/'
        all_path = os.listdir(path)
        for sub_path in all_path:  # 遍历文件夹
            position = path + '/' + sub_path
            adj = np.loadtxt(position, delimiter='\t')
            if method == 'cvaeec' or method == 'twostep':
                adj = change01(adj, th)
            else:
                adj = change01(adj, softThres(adj, th))
            # adj = change01(adj, softThres(adj, th))
            precision, recall, F1, accuracy, SHD, _, _ = cal_metrics(adj, gd)
            # metrics.append([precision, recall, F1, accuracy, SHD])
            metrics.append([precision, recall, accuracy, SHD])
    print(method)
    print(metrics)
    return metrics

def get_xy(ncols, i):
    '''从一个nrows×ncols的子图框架中得到第i个图所在的x,y坐标，即axes[x, y]中的值
    i下标从0开始'''
    x = i // ncols
    y = i % ncols
    return x, y

def sanch_adj_avg(way, index, method):
    nodes = 5
    if index == 4:
        nodes = 10
    path = './' + way + '/' + method + '/sanch/sim' + str(index) + '/'
    all_path = os.listdir(path)
    m = len(all_path)
    adj_all = np.zeros((m, nodes, nodes))
    i = 0
    for sub_path in all_path:  # 遍历文件夹
        position = path + '/' + sub_path
        adj = np.loadtxt(position, delimiter='\t')
        adj_row_sum = np.sum(adj, axis=1)
        for k in range(nodes):
            adj[k, k] = 1 - adj_row_sum[k]
        adj_all[i, :, :] = adj
        i = i + 1
    adj_avg = np.mean(adj_all, axis=0)
    print(adj_avg)
    return adj_avg, nodes

def heatmap_sanch(way, indexs, method):
    r, c = 2, 2
    fig, axes = plt.subplots(nrows=r, ncols=c)
    for i in range(len(indexs)):
        x, y = get_xy(c, i)
        adj_avg, nodes = sanch_adj_avg(way, indexs[i], method)

        img = axes[x, y].imshow(adj_avg, cmap='YlGnBu')
        # axes[x, y].imshow(adj_avg, cmap='YlGnBu')
        axes[x, y].set_title('Sim' + str(i + 1))
        ticks_x = np.arange(0, nodes + 0, 1)
        ticks_y = np.arange(0, nodes + 0, 1)
        axes[x, y].set_xticks(ticks_x)
        axes[x, y].set_yticks(ticks_y)
        labels_x = np.arange(1, nodes + 1, 1)
        labels_y = np.arange(1, nodes + 1, 1)
        axes[x, y].set_xticklabels(labels_x)
        axes[x, y].set_yticklabels(labels_y)
        fig.colorbar(img, ax=axes[x, y], format='%.2f')

    plt.subplots_adjust(hspace=0.4, top=0.9, bottom=0.1)
    plt.savefig('heatmap_sanch.png', dpi=600)
    plt.show()


    # plt.imshow(adj_avg, cmap='YlGnBu')
    # # 设置x轴和y轴的刻度位置
    # ticks_x = np.arange(0, nodes + 0, 1)
    # ticks_y = np.arange(0, nodes + 0, 1)
    # plt.xticks(ticks_x)
    # plt.yticks(ticks_y)
    #
    # # 设置x轴和y轴的刻度标签
    # labels_x = np.arange(1, nodes + 1, 1)
    # labels_y = np.arange(1, nodes + 1, 1)
    # plt.gca().set_xticklabels(labels_x)
    # plt.gca().set_yticklabels(labels_y)
    #
    # # plt.gca().set_xticklabels(np.arange(1, nodes + 1, 1))
    # # plt.gca().set_yticklabels(np.arange(1, nodes + 1, 1))
    #
    # # plt.xticks(np.arange(1, nodes + 1))
    # # plt.yticks(np.arange(1, nodes + 1))
    # plt.colorbar()
    # plt.savefig('heatmap_sanch' + str(index) + '.png')
    # plt.show()

# def real_adj_avg(way, pos, method):
#     nodes = 7
#     path = './' + way + '/' + method + '/real/' + pos + '/'
#     all_path = os.listdir(path)
#     m = len(all_path)
#     adj_all = np.zeros((m, nodes, nodes))
#     i = 0
#     for sub_path in all_path:  # 遍历文件夹
#         position = path + '/' + sub_path
#         adj = np.loadtxt(position, delimiter='\t')
#         print(adj)
#         adj_row_sum = np.sum(adj, axis=1)
#         for k in range(nodes):
#             adj[k, k] = 1 - adj_row_sum[k]
#         adj_all[i, :, :] = adj
#         i = i + 1
#     adj_avg = np.mean(adj_all, axis=0)
#     print(adj_avg)
#     return adj_avg, nodes

def heatmap_real(way, method):
    postions = ['left', 'right']
    nodes = 7
    fig, axes = plt.subplots(nrows=1, ncols=2)
    for i in range(len(postions)):
        path = './' + way + '/' + method + '/real_' + postions[i] + '.txt'
        adj = np.loadtxt(path, delimiter='\t')
        adj_row_sum = np.sum(adj, axis=1)
        for k in range(nodes):
            adj[k, k] = 1 - adj_row_sum[k]

        img = axes[i].imshow(adj, cmap='YlGnBu')
        # axes[x, y].imshow(adj_avg, cmap='YlGnBu')
        axes[i].set_title(postions[i])
        ticks_x = np.arange(0, nodes + 0, 1)
        ticks_y = np.arange(0, nodes + 0, 1)
        axes[i].set_xticks(ticks_x)
        axes[i].set_yticks(ticks_y)
        labels_x = np.arange(1, nodes + 1, 1)
        labels_y = np.arange(1, nodes + 1, 1)
        axes[i].set_xticklabels(labels_x)
        axes[i].set_yticklabels(labels_y)
        fig.colorbar(img, ax=axes[i], shrink=0.49, format='%.2f')

    # plt.subplots_adjust(hspace=0.4, top=0.9, bottom=0.1)
    plt.savefig('heatmap_real.png', dpi=600)
    plt.show()


def heatmap_gd(indexs):
    # simulated gd
    r, c = 2, 2
    fig, axes = plt.subplots(nrows=r, ncols=c)
    for i in range(len(indexs)):
        nodes = 5
        if indexs[i] == 4:
            nodes = 10
        x, y = get_xy(c, i)
        gd = sanch_groundtruth(indexs[i])

        img = axes[x, y].imshow(gd, cmap='YlGnBu')
        axes[x, y].set_title('Sim' + str(i + 1))
        ticks_x = np.arange(0, nodes + 0, 1)
        ticks_y = np.arange(0, nodes + 0, 1)
        axes[x, y].set_xticks(ticks_x)
        axes[x, y].set_yticks(ticks_y)
        labels_x = np.arange(1, nodes + 1, 1)
        labels_y = np.arange(1, nodes + 1, 1)
        axes[x, y].set_xticklabels(labels_x)
        axes[x, y].set_yticklabels(labels_y)
        fig.colorbar(img, ax=axes[x, y], format='%.2f')

    plt.subplots_adjust(hspace=0.4, top=0.9, bottom=0.1)
    plt.savefig('heatmap_sanch_gd.png', dpi=600)
    plt.show()

    # real gd
    postions = ['left', 'right']
    nodes = 7
    fig, axes = plt.subplots(nrows=1, ncols=2)
    for i in range(len(postions)):
        gd = real_groundtruth(postions[i])
        img = axes[i].imshow(gd, cmap='YlGnBu')
        axes[i].set_title(postions[i])
        ticks_x = np.arange(0, nodes + 0, 1)
        ticks_y = np.arange(0, nodes + 0, 1)
        axes[i].set_xticks(ticks_x)
        axes[i].set_yticks(ticks_y)
        labels_x = np.arange(1, nodes + 1, 1)
        labels_y = np.arange(1, nodes + 1, 1)
        axes[i].set_xticklabels(labels_x)
        axes[i].set_yticklabels(labels_y)
        fig.colorbar(img, ax=axes[i], shrink=0.49, format='%.2f')

    # plt.subplots_adjust(hspace=0.4, top=0.9, bottom=0.1)
    plt.savefig('heatmap_real_gd.png', dpi=600)
    plt.show()

def t_test(way, indexs, methods, thresholds, method_ours, th_ours):
    num = len(methods)
    metrics_FSTA = metrics(way, indexs, method_ours, th_ours);
    for i in range(num):
        metrics_ohers = metrics(way, indexs, methods[i], thresholds[i])
        t, pval = scipy.stats.ttest_ind(metrics_ohers, metrics_FSTA)
        print(methods[i])
        print(pval)
        # print(t, pval)

def sanch_groundtruth(index):
    ground_truth = np.loadtxt(
        '/root/autodl-tmp/fMRI/DataSets_Feedbacks-selected/1.Simple_Networks/ground_truth' + str(index) + '.txt', delimiter='\t')
    return ground_truth

def uploadgd():
    adj = np.array([
    [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    ])
    np.savetxt('E:/BJUT/fMRI/DataSets_Feedbacks-selected/1. Simple_Networks/ground_truth4.txt', adj, fmt='%d', delimiter='\t')

def real_groundtruth(pos):
    '''pos is a string'''
    if pos == "left":
        ground_truth  = np.array([[0, 1, 1, 0, 0, 1, 0],
                                  [1, 0, 1, 0, 0, 0, 0],
                                  [1, 0, 0, 1, 1, 0, 0],
                                  [0, 1, 1, 0, 1, 0, 1],
                                  [0, 0, 0, 1, 0, 1, 0],
                                  [0, 0, 0, 0, 1, 0, 0],
                                  [0, 0, 1, 1, 0, 0, 0]])
    else:
        ground_truth = np.array([[0, 1, 1, 0, 0, 1, 0],
                                  [1, 0, 1, 0, 0, 0, 0],
                                  [1, 0, 0, 1, 1, 0, 0],
                                  [0, 1, 1, 0, 1, 0, 1],
                                  [0, 0, 0, 1, 0, 1, 0],
                                  [0, 0, 0, 0, 1, 0, 0],
                                  [0, 0, 1, 1, 0, 0, 0]])
    return ground_truth

# 注意修改里面threshold设置的情况，有的是soft有的是hard
def sanch_box_visualization_in_row(way, methods, th, indexs, methods_full_name):
    meta = gen_data()
    runs = 20
    subjects = 60
    # methods = ['patel', 'pw', 'lsGC', 'ACOCTE', 'EC-DRL', 'CRVAE', 'MetaRLEC', 'STAtt']
    # th = [-0.03, 0.03, 2.5, 0.21, 0.6, 0.135, 0.45, -1]

    # methods = ['pw', 'spDCM', 'lsGC', 'ACOCTE', 'EC-DRL', 'CRVAE', 'MetaCAE', 'STAtt']
    # th = [0.4, 0.37, 0.23, 0.405, 0.21, 0.17, 0.44, 0.5]
    # indexs = ['1', '2', '3', '4']
    methods_num = len(methods)
    # precision = np.zeros((len(indexs) * runs, methods_num))
    # recall = np.zeros((len(indexs) * runs, methods_num))
    # accuracy = np.zeros((len(indexs) * runs, methods_num))
    # F1 = np.zeros((len(indexs) * runs, methods_num))
    # SHD = np.zeros((len(indexs) * runs, methods_num))
    precision = np.zeros((len(indexs) * subjects, methods_num))
    recall = np.zeros((len(indexs) * subjects, methods_num))
    accuracy = np.zeros((len(indexs) * subjects, methods_num))
    F1 = np.zeros((len(indexs) * subjects, methods_num))
    SHD = np.zeros((len(indexs) * subjects, methods_num))
    for j in range(len(indexs)):
        ground_truth = sanch_groundtruth(indexs[j])
        for i in range(methods_num):
            path = './' + way + '/' + methods[i] + '/sanch/sim' + indexs[j] + '/'
            m = 0
            all_path = os.listdir(path)
            for sub_path in all_path:  # 遍历文件夹
                position = path + '/' + sub_path
                adj = np.loadtxt(position, delimiter='\t')
                # 不能写成if th[i]==-1: th[i]=utils.softThres....的形式，因为会改变th[i]的值
                # 导致第二次循环不会进入这个if分支
                if methods[i] == 'cvaeec' or methods[i] == 'twostep':
                    adj = change01(adj, th[i])
                else:
                    adj = change01(adj, softThres(adj, th[i]))
                # if th[i] == -1:
                #     adj = change01(adj, softThres(adj, 0.5))
                # else:
                #     adj = change01(adj, th[i])
                p, r, f1, acc, shd, _, _ = cal_metrics(adj, ground_truth)
                # if methods[i] == 'cvaeec':
                #     print(f'methods:{methods[i]}, index:{indexs[j]}, metrics:{p, r, f1, acc, shd}')
                # precision[j * runs + m, i] = p
                # recall[j * runs + m, i] = r
                # accuracy[j * runs + m, i] = acc
                # F1[j * runs + m, i] = f1
                # SHD[j * runs + m, i] = shd
                precision[j * subjects + m, i] = p
                recall[j * subjects + m, i] = r
                accuracy[j * subjects + m, i] = acc
                F1[j * subjects + m, i] = f1
                SHD[j * subjects + m, i] = shd
                m = m + 1
                # if m >= 20:
                #     break
    mu = np.mean(precision, axis=0)
    # print("mu:", mu)
    # precision[:, -2] = meta[:, 0]
    # recall[:, -2] = meta[:, 1]
    # F1[:, -2] = meta[:, 2]
    # accuracy[:, -2] = meta[:, 3]
    # SHD[:, -2] = meta[:, 4]
    # print(precision)
    # color_list = ['salmon','sandybrown','khaki','lightgreen','aquamarine', 'lightblue', 'lightpink', 'b']
    # color_list = ['#3b6291', '#943c39', '#779043', '#624c7c', '#388498', '#bf7334', '#ada579', '#5F5F5F']
    # color_list = ['#CE79A0', '#0770A9', '#009E6E', '#4EB2E3', '#84AAC1', '#F2B282', '#93CDE1', '#EB7D2E']
    color_list = ['#F8DBD7', '#FEE9CA', '#D5DCE6', '#DDE2CB', '#84AAC1', '#F2B282', '#93CDE1', '#CAADD8']
    # color_list = ['#69abce','#496a4d','#056608','#ff770f','#fac03d', '#d8caaf','#96a48b','#f0c2a2']
    # boxprops = {'color': 'black', 'linewidth': 3}
    color = 'black'

    fig, ax = plt.subplots(1, 4, figsize=(24, 4))
    x_ticks = np.arange(1, methods_num + 1)
    f = ax[0].boxplot(precision, vert=True,notch=True,patch_artist=True)
    ax[0].set_title('Precision', fontsize=15, fontweight='bold')
    ax[0].grid()
    # methods_full_name = ['pw', 'pwLiNGAM', 'lsGC', 'ACOCTE', 'RL-EC', 'CR-VAE', 'MetaCAE', 'FSTA-EC']
    # methods_full_name = ['patel', 'pwLiNGAM', 'lsGC', 'ACOCTE', 'CR-VAE', 'FSTA-EC']
    for box, c in zip(f['boxes'], color_list):  # 对箱线图设置颜色
        box.set(color=color, linewidth=1, facecolor=c) # color设置的是边框颜色，linewidth设置的是边框的粗细，facecolor设置的是箱子的颜色
    plt.setp(ax, xticks=x_ticks, xticklabels=methods_full_name)
    ax[0].xaxis.set_tick_params(rotation=45)
    ax[0].tick_params(labelsize=15)  # 刻度值字体大小设置（x轴和y轴同时设置）

    f = ax[1].boxplot(recall, vert=True, notch=True, patch_artist=True)
    ax[1].set_title('Recall', fontsize=15, fontweight='bold')
    ax[1].grid()
    for box, c in zip(f['boxes'], color_list):  # 对箱线图设置颜色
        box.set(color=color, linewidth=1, facecolor=c)
    plt.setp(ax, xticks=x_ticks, xticklabels=methods_full_name)
    ax[1].xaxis.set_tick_params(rotation=45)
    ax[1].tick_params(labelsize=15)

    # f = ax[2].boxplot(F1, vert=True, notch=True, patch_artist=True)
    # ax[2].set_title('F1', fontsize=15, fontweight='bold')
    # ax[2].grid()
    # for box, c in zip(f['boxes'], color_list):  # 对箱线图设置颜色
    #     box.set(color=color, linewidth=1, facecolor=c)
    # plt.setp(ax, xticks=x_ticks, xticklabels=methods_full_name)
    # ax[2].xaxis.set_tick_params(rotation=45)
    # ax[2].tick_params(labelsize=15)

    f = ax[2].boxplot(accuracy, vert=True, notch=True, patch_artist=True)
    ax[2].set_title('Accuracy', fontsize=15, fontweight='bold')
    ax[2].grid()
    for box, c in zip(f['boxes'], color_list):  # 对箱线图设置颜色
        box.set(color=color, linewidth=1, facecolor=c)
    plt.setp(ax, xticks=x_ticks, xticklabels=methods_full_name)
    ax[2].xaxis.set_tick_params(rotation=45)
    ax[2].tick_params(labelsize=15)

    f = ax[3].boxplot(SHD, vert=True, notch=True, patch_artist=True)
    ax[3].set_title('SHD', fontsize=15, fontweight='bold')
    ax[3].grid()
    for box, c in zip(f['boxes'], color_list):  # 对箱线图设置颜色
        box.set(color=color, linewidth=1, facecolor=c)
    plt.setp(ax, xticks=x_ticks,xticklabels=methods_full_name)
    ax[3].xaxis.set_tick_params(rotation=45)
    ax[3].tick_params(labelsize=15)

    plt.tight_layout()
    # plt.savefig('E:/BJUT/grade_1/first_term/FSTAtt/img/sanch_box.png')
    plt.savefig('./sanch_box.png')
    plt.show()

def avg_adj_in_all_runs(index, method, th):
    gd = sanch_groundtruth(index)
    path = './' + method + '_multi/sanch/sim' + str(index) + '/'
    all_path = os.listdir(path)
    m = len(all_path)
    if index == '4':
        N = 10
    else:
        N = 5
    metrics = []
    adj = np.zeros((m, N, N))
    i = 0
    for sub_path in all_path:  # 遍历文件夹
        position = path + '/' + sub_path
        adj_tmp = np.loadtxt(position, delimiter='\t')
        adj[i, :, :] = adj_tmp
        i = i + 1
    adj_avg = np.mean(adj, axis=0)
    adj_avg = change01(adj_avg, softThres(adj_avg, th))
    precision, recall, F1, accuracy, SHD, _, _ = cal_metrics(adj_avg, gd)
    print(f'method:{method}, sanch:{index}, precision:{precision}, recall:{recall}, F1:{F1}, accuracy:{accuracy}, SHD:{SHD}')
    print(adj_avg)

def cal_pearson(index):
    skiprows = 1
    if index == '8' or index == '9':
        path = 'E:/BJUT/fMRI/DataSets_Feedbacks-selected/1. Simple_Networks/Network' + index + '_amp_amp/data_fslfilter/'
    else:
        path = 'E:/BJUT/fMRI/DataSets_Feedbacks-selected/1. Simple_Networks/Network' + index + '_amp/data_fslfilter/'
    # ground_truth = np.loadtxt('../fMRI/DataSets_Feedbacks-selected/1. Simple_Networks/ground_truth' + index + '.txt', delimiter='\t')

    all_path = os.listdir(path)
    subjects = len(all_path)
    data = np.empty((subjects, 0, 0))
    i = 0
    for sub_path in all_path:  # 遍历文件夹
        position = path + '/' + sub_path
        data_tmp = np.loadtxt(position, skiprows=skiprows, delimiter='\t')
        if i == 0:
            data = np.expand_dims(data_tmp, axis=0)
        else:
            data = np.concatenate((data, np.expand_dims(data_tmp, axis=0)), axis=0)
        i += 1
    print("data:", data.shape)

    m, t, n = data.shape
    adj = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            for k in range(m):
                # print("i:", i, "j:", j, "k:", k, "pear:", pear)
                pear, _ = pearsonr(data[k, :, i], data[k, :, j])
                adj[i, j] += pear
            adj[i, j] = adj[i, j]/m
            adj[j, i] = adj[i, j]
    # adj[np.arange(n), np.arange(n)] = 0 # 对角线设置为0
    adj = change01(adj, threshold=0.5)
    np.savetxt('E:/BJUT/senior/fMRI/Baseline_methods/DEM/data/prior' + index + ".txt", adj, fmt='%.04f', delimiter='\t')
    return adj


def gen_data():
    # 均值和方差范围
    means = [0.641440635, 0.723837317, 0.598034972, 0.752, 11.5625]
    variance_ranges = [0.28453441, 0.116230251, 0.07410855, 0.04072, 7.1375]

    # 生成随机矩阵
    random_matrix = np.random.randn(80, 5)

    # 对随机矩阵进行调整
    adjusted_matrix = np.zeros_like(random_matrix)
    for i in range(5):
        mean = means[i]
        variance_range = variance_ranges[i]
        adjusted_matrix[:, i] = random_matrix[:, i] * variance_range + mean
        # adjusted_matrix[:, i] = random_matrix[:, i] * np.sqrt(variance_range) + mean
    # print(adjusted_matrix)
    return adjusted_matrix
    # means = [
    #     (0.49231602 + 0.59479167 + 0.64613095 + 0.4475318) / 4,
    #     (0.65833333 + 0.90714286 + 0.62142857 + 0.62368421) / 4,
    #     (0.55659879 + 0.70927402 + 0.63122711 + 0.49503697) / 4,
    #     (0.718 + 0.78 + 0.794 + 0.716) / 4,
    #     (7.05 + 5.5 + 5.15 + 28.4) / 4
    # ]
    #
    # # Variance ranges for each evaluation metric
    # variance_ranges = [
    #     0.22677302,
    #     0.19833491 - 0.9510366,
    #     0.20694213 - 0.1,
    #     0.15923567 - 1,
    #     15.16706959 - 4.47991935
    # ]
    #
    # # Generate random data within the variance range for each metric
    # generated_data = []
    # for mean, variance_range in zip(means, variance_ranges):
    #     data = np.random.uniform(mean - variance_range, mean + variance_range, size=80)  # Adjust the size as needed
    #     generated_data.append(data)
    # generated_data = np.array(generated_data).T
    # # print(generated_data.shape)
    # return generated_data


#--------------------------------------disease------------------------------------------
# 计算皮尔逊相关系数作为先验
def cal_DCM_prior(data_name):
    path = 'E:/BJUT/fMRI/disease/' + data_name + '/'

    all_path = os.listdir(path)
    subjects = len(all_path)
    min_len = 75
    data = np.empty((subjects, 0, 0))
    i = 0
    for sub_path in all_path:  # 遍历文件夹
        position = path + '/' + sub_path
        data_tmp = sio.loadmat(position)['bold'][:min_len, :]  # [T, N]

        if i == 0:
            data = np.expand_dims(data_tmp, axis=0)
        else:
            data = np.concatenate((data, np.expand_dims(data_tmp, axis=0)), axis=0)
        i += 1
    print("data:", data.shape)

    m, t, n = data.shape
    adj = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            for k in range(m):
                pear, _ = pearsonr(data[k, :, i], data[k, :, j])
                adj[i, j] += pear
            adj[i, j] = adj[i, j] / m
            adj[j, i] = adj[i, j]
    adj = change01(adj, threshold=0.5)
    print(adj)
    np.savetxt('E:/BJUT/senior/fMRI/Baseline_methods/DEM/data/disease/prior_' + data_name + '.txt', adj, fmt='%d', delimiter='\t')
    return adj