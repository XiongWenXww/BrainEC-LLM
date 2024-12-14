import argparse
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import BrainECLLM
from data_provider.data_loader import prepare_dataloaders
from utils.postprocess import softThres, change01, change01_constraint, change01_constraint_child
from utils.metrics import adj_metrics, metric
from utils.Optim import ScheduledOptim
from utils.tools import del_files, EarlyStopping, adjust_learning_rate, load_content, adj_description, matrix_to_text, seed_everything
from utils.losses import loss_func

import time
import numpy as np
import os
import copy

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

torch.set_printoptions(precision=3) # set the print data precision

parser = argparse.ArgumentParser(description='BrainEC-LLM')

# basic config
parser.add_argument('--seed', type=int, default=2021, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0', help='training device')
parser.add_argument('--model', type=str, default='BrainEC-LLM',
                    help='model name, options: [BrainEC-LLM]')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
parser.add_argument('--ablation', type=str, default='no', help='ablation study')

# data loader
parser.add_argument('--data', type=str, required=True, default='Smith', help='options:[Smith, Sanchez]')
parser.add_argument('--index', type=int, default=1, help='the index in the simulated dataset')
parser.add_argument('--pos', type=str, default='left', help='the position in the real dataset')
parser.add_argument('--path', type=str, default='../fMRI/sims_txt/', help='path of the data file')
parser.add_argument('--is_norm', type=bool, default=False, help='use normalization')
parser.add_argument('--sliding_window', type=int, default=0, help='options:[0:do not use sliding window, 1:use sliding window]')
parser.add_argument('--window_size', type=int, default=200, help='the window size of the sliding window method')
parser.add_argument('--overlap', type=int, default=5, help='the overlap of the sliding window method')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# optimization
parser.add_argument('--lower_bound', type=float, required=True, default=0.6, help='Lower bound for connection strength')
parser.add_argument('--upper_bound', type=float, required=True, default=0.8, help='Upper bound on the strength of the connection')
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--shuffle', type=bool, default=True, help='data loader shuffle')
parser.add_argument('--epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=500, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--percent', type=int, default=100)

parser.add_argument('-warmup','--n_warmup_steps', type=int, default=4000, help='ScheduledOptim parameter')
parser.add_argument('-lr_mul', type=float, default=1.2, help='ScheduledOptim parameter')

# lora
parser.add_argument('--r', type=int, default=64, help='lora attention dimension(the rank)')
parser.add_argument('--lora_alpha', type=int, default=32)
parser.add_argument('--lora_dropout', type=float, default=0.1)

# model define
# main(LLM)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--ts_len', type=float, default=None, help='time series length')
parser.add_argument('--nodes_num', type=float, default=None, help='number of brain region')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn in cross attention')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=3, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--llm_model', type=str, default='LLAVA-MED', help='LLM model, options:[LLAMA3, LLAMA, GPT2, BERT, LLAVA-MED]')
parser.add_argument('--llm_path', type=str, default='llava-med-v1.5-mistral-7b', help='LLM model path')
parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768

## Mutlitscale
parser.add_argument('--channel_independence', type=int, default=0,
                    help='0: channel dependence. 1: the channel dim is 1, in this case, N=1')
parser.add_argument('--down_sampling_method', type=str, default='avg', help='in the first stage, options:[max, avg, conv] or others')
parser.add_argument('--down_sampling_layers', type=int, default=3, help='the layers num i corresponds to (i+1) multiscale input series')
parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
parser.add_argument('--decomp_method', type=str, default='moving_avg', help='in the past decomposable mixing, \
                    method of series decompsition, only support moving_avg or dft_decomp')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, \
                    t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
## patch embedding
# The following conditions must be met: T_min+stride > (T-patch_len)/stride+2, T_min:The smallest scale of data
parser.add_argument('--patch_len', type=int, default=8, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')

# GroupFFN
parser.add_argument('--kernel_size', nargs='+',type=int, default=5, help='kernel size')
parser.add_argument('--MS_d_ff', type=int, default=32, help='dimension of fcn')
## VGAE
parser.add_argument('--VAGE_in_dim', type=int, default=None, help='input dim in VGAE, should be the same with the last dim of data in GCN')
parser.add_argument('--VAGE_hidden1_dim', type=int, default=32, help='hidden layer1 dim in VGAE')

## Att
parser.add_argument('--att_hidden', type=int, default=16, help='hidden layer in FFN of Attention')
parser.add_argument('--n_head', type=int, default=2, help='head number')
parser.add_argument('--d_k', type=int, default=None, help='d_k in Attetnion, d_model=n_head * d_k(Query and Key), d_model \
                    is the last dim of data, in this case, data first transpose, then d_mode=T')
parser.add_argument('--d_v', type=int, default=None, help='d_v in Attetnion, d_model=n_head * d_v(Value)')

# loss
parser.add_argument('--proportion', type=float, default=0.4, help='the proportion of maximum nodes in each row and column')
parser.add_argument('--max_parent_proportion', type=float, default=0.4, help='the proportion of maximum parent nodes')
parser.add_argument('--max_child_proportion', type=float, default=0.4, help='the proportion of maximum child nodes')
parser.add_argument('--soft_threshold', type=float, default=0.5, help='soft threshold of adj')
parser.add_argument('--alpha_sp', type=float, default=0.8, help='soft threshold hyperparameter')
parser.add_argument('--alpha_DAG', type=float, default=0.8, help='DAG constraint hyperparameter')
parser.add_argument('--alpha_cons', type=float, default=0, help='constrastive loss hyperparameter')
parser.add_argument('--temperature', type=float, default=0.2, help='temperature value in contrastive loss')

args = parser.parse_args()
accelerator = None
print('cuda available:', torch.cuda.is_available())

seed_everything(args.seed)

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}{}_llm_layers{}_dropout{}_dm{}_nh{}_el{}_df{}_fc{}_eb{}_{}'.format(
        args.data,
        args.index,
        args.llm_layers,
        args.dropout,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_ff,
        args.factor,
        args.embed, ii)
    print('setting:', setting)

    data_loader, ground_truth = prepare_dataloaders(args)
    args.d_v = args.d_k = int(args.ts_len/args.n_head)
    print(ground_truth)

    content = load_content(args)
    args.content = content
    print('content:', args.content)
    args.adj_text = matrix_to_text(args.adj)
    print(args.adj_text)

    args.adj = torch.FloatTensor(args.adj).to(args.device)

    model = BrainECLLM.Model(args).float()
    model = model.to(args.device)

    time_now = time.time()
    time_all = time.time()

    data_steps = len(data_loader)
    print('data_steps:', data_steps)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)
    # model_optim = ScheduledOptim(
    #     optim.Adam([{'params': trained_parameters}], betas=(0.9, 0.98), eps=1e-09),
    #     lr_mul=args.lr_mul, d_model=args.ts_len, n_warmup_steps=args.n_warmup_steps)
    
    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=data_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.epochs,
                                            max_lr=args.learning_rate)

    criterion = loss_func(alpha_sp=args.alpha_sp, alpha_DAG=args.alpha_DAG).to(args.device)
    # mae_metric = nn.L1Loss()

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    best_SHD = 10000
    best_F1 = 0
    for epoch in range(args.epochs):
        iter_count = 0
        train_loss = []
        adj = []
        weight = []

        model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y) in tqdm(enumerate(data_loader)):
            iter_count += 1
            model_optim.zero_grad()

            batch_x = batch_x.float().to(args.device)
            batch_y = batch_y.float().to(args.device)

            if args.use_amp:
                with torch.cuda.amp.autocast(): # 是PyTorch中一种混合精度的技术（仅在GPU上训练时可使用），可在保持数值精度的情况下提高训练速度和减少显存占用
                    if args.output_attention:
                        L_constrastive, pred_y, pred_adj = model(batch_x)[0]
                    else:
                        L_constrastive, pred_y, pred_adj = model(batch_x)
                    loss = criterion(pred_y, pred_adj, batch_y, args.soft_threshold)+L_constrastive
                    train_loss.append(loss.item())
                    adj.append(pred_adj.cpu().detach().numpy())
                    weight.append(pred_y.shape[0])
            else:
                if args.output_attention:
                    L_constrastive, pred_y, pred_adj = model(batch_x)[0]
                else:
                    L_constrastive, pred_y, pred_adj = model(batch_x)
                loss = criterion(pred_y, pred_adj, batch_y, args.soft_threshold)+args.alpha_cons*L_constrastive
                train_loss.append(loss.item())
                adj.append(pred_adj.cpu().detach().numpy())
                weight.append(pred_y.shape[0])

            if (i + 1) % 100 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.epochs - epoch) * data_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                loss.backward()
                model_optim.step()

            if args.lradj == 'TST':
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()

        train_loss = np.average(train_loss, weights=weight)
        adj_mean = np.average(adj, weights=weight, axis=0) # [c, N, N]->[N, N], c is the number of total batch num, that is S/batch_size

        adj_init = copy.deepcopy(adj_mean)
        np.fill_diagonal(adj_mean, 0) # 先将对角线置为0再求soft threshold，以免对角线的值会影响soft threshold的结果
        threshold = softThres(adj_mean, args.soft_threshold)
        adj_binary = change01_constraint(adj_mean, threshold, args.proportion)
        precision, recall, F1, accuracy, SHD = adj_metrics(adj_binary, ground_truth)
        if SHD < best_SHD:
            best_SHD = SHD
        if F1 > best_F1:
            best_F1 = F1
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f'row->col, data:{args.data}, index:{args.index}, epoch:{epoch + 1}, loss:{train_loss: .3f}, cost time:{time.time() - epoch_time:.3f}, best_SHD:{best_SHD}, best_F1:{best_F1:.3f}')
            print(f'threshold:{threshold:.3f}, precision:{precision:.3f}, recall:{recall:.3f}, F1:{F1:.3f}, accuracy:{accuracy:.3f}, SHD:{SHD}')
            print(adj_binary)

        early_stopping(train_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        if args.lradj != 'TST':
            if args.lradj == 'COS':
                scheduler.step()
                print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                if epoch == 0:
                    args.learning_rate = model_optim.param_groups[0]['lr']
                    print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)
        else:
            print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
    print(f'time all:{time.time() - time_all: .3f}')
    