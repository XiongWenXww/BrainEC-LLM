# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""
import torch
import torch as t
import torch.nn as nn
import numpy as np
import pdb

class loss_func(nn.Module):
    def __init__(self, alpha_sp, alpha_DAG):
        super(loss_func, self).__init__()
        self.alpha_sp = alpha_sp
        self.criterion = nn.MSELoss()
        self.alpha_DAG = alpha_DAG
        # self.alpha_acy = alpha_acy

    def forward(self, output, adj, label, soft_threshold):
        # attention is all xx:loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
        # adj_init = copy.deepcopy(adj)

        # adj = adj - torch.diag(torch.diag(adj)) # set the diagonal values to 0
        # threshold = softThres(adj, soft_threshold)
        # adj_binary = change01GPU(adj, threshold=threshold)
        # L_sp = torch.sum(torch.sum(adj_binary)) # From the experimental results, it can be seen that this does not effectively control sparsity

        L_sp = torch.sum(torch.sum(adj))

        # q_probs = F.softmax(output, dim=-1)
        # p_probs = F.softmax(label, dim=-1)
        
        # # 添加一个小的常数以避免 log(0) 的情况
        # epsilon = 1e-9
        # q_probs = q_probs.clamp(min=epsilon, max=1 - epsilon)
        
        # cross_entropy = -torch.sum(p_probs * torch.log(q_probs), dim=-1)
        # L_re = cross_entropy.mean(dim=0).mean(dim=0)

        # 原时域损失
        # L_re = ((output-label)**2).mean()
        criterion = nn.MSELoss()
        L_re = criterion(output, label)
        # 所提频域损失
        # L_feq = (torch.fft.rfft(output, dim=1) - torch.fft.rfft(label, dim=1)).abs().mean() 

        # L_pre = F.cross_entropy(output, label)
        # L_pre = self.criterion(output, label)
        # print('L_re:', L_re, 'L_sp:', L_sp, 'L_feq:', L_feq)
        # return L_re + self.alpha_sp * L_sp + self.alpha_sp * L_feq
        # print('L_re:', L_re, 'L_sp:', L_sp)
        L_DAG = dag_loss(adj)
        # print('L_re:', L_re, 'L_sp:', L_sp, 'L_DAG:', L_DAG)
        return L_re + self.alpha_sp * L_sp + self.alpha_DAG * L_DAG
        # return L_re

def dag_loss(W):
    """
    Compute the DAG loss for a given adjacency matrix W.

    Args:
        W (torch.Tensor): Adjacency matrix (n x n).

    Returns:
        torch.Tensor: The DAG loss.
    """
    n = W.shape[0]
    element_wise_product = W * W # Compute element-wise product W ∘ W
    expm = torch.matrix_exp(element_wise_product)
    trace = torch.trace(expm)

    # Compute the DAG loss
    loss = trace - n

    return loss

def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)
