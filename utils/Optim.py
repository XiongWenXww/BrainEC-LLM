'''A wrapper class for scheduled optimizer '''
import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from utils.tools import EarlyStopping
from utils.losses import loss_func

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0


    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


def init_optim(args, accelerator, model):
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
                                            steps_per_epoch=1, # len(data_loader)
                                            pct_start=args.pct_start,
                                            epochs=args.epochs,
                                            max_lr=args.learning_rate)

    criterion = loss_func(alpha_sp=args.alpha_sp, alpha_DAG=args.alpha_DAG).to(args.device)
    # mae_metric = nn.L1Loss()

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        return early_stopping, model_optim, scheduler, criterion, scaler
    else:
        return early_stopping, model_optim, scheduler, criterion