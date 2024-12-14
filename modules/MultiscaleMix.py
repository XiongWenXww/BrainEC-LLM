import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class MultiscaleMix(nn.Module):
    def __init__(self, args):
        super(MultiscaleMix, self).__init__()
        self.channel_independence = args.channel_independence
        self.ts_len = args.ts_len
        self.temperature = args.temperature

        self.predict_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    self.ts_len // (args.down_sampling_window ** i),
                    self.ts_len,
                )
                for i in range(args.down_sampling_layers + 1)
            ]
        )

        self.project_head_layers_i = torch.nn.ModuleList(
            [
                ProjectionHead(
                        input_dim=args.ts_len // (args.down_sampling_window ** i), # T_i
                        hidden_dim=args.ts_len // (args.down_sampling_window ** i), # T_i
                        output_dim=args.ts_len // (args.down_sampling_window ** i) # T_i
                    )
                    for i in range(args.down_sampling_layers)
            ]
        )

        self.project_head_layers_i1 = torch.nn.ModuleList(
            [
                ProjectionHead(
                        input_dim=args.ts_len // (args.down_sampling_window ** (i + 1)), # T_{i+1}
                        hidden_dim=args.ts_len // (args.down_sampling_window ** i), # T_i
                        output_dim=args.ts_len // (args.down_sampling_window ** i) # T_i
                    )
                    for i in range(args.down_sampling_layers)
            ]
        )

        self.project_head_layers_ij = nn.ModuleList([
            nn.ModuleList([
                ProjectionHead(input_dim=args.ts_len // (args.down_sampling_window ** j), # T_j
                        hidden_dim=args.ts_len // (args.down_sampling_window ** i), # T_i
                        output_dim=args.ts_len // (args.down_sampling_window ** i) # T_i
                        ) 
                for j in range(args.down_sampling_layers + 1)]) 
            for i in range(args.down_sampling_layers)])

    def forward(self, fine_scale_list):
        # [multiscale, B, T_i, N], [multiscale, B, T_i, N]
        L_constrastive = self.calc_L_constrastive(fine_scale_list, self.temperature)
        # Future Multipredictor Mixing as decoder for future
        dec_out_list = self.future_multi_mixing(fine_scale_list) # [multiscale, B, T, N]

        # list size:multiscale, list content: [B, T, N]->[B, T, N, multiscale]->[B, T, N]
        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        return L_constrastive, dec_out # value, [B, T, N]
    
    def future_multi_mixing(self, fine_scale_list):
        # [multiscale, B, T_i, N]
        dec_out_list = []
        if self.channel_independence == 1:
            for i, enc_out in zip(range(len(fine_scale_list)), fine_scale_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension, [B, T_i, N]->[B, T, N]
                dec_out_list.append(dec_out)
        else:
            for i, enc_out in zip(range(len(fine_scale_list)), fine_scale_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension, [B, T_i, N]->[B, T, N]
                dec_out_list.append(dec_out)

        return dec_out_list # value, [multiscale, B, T, N]
    
    def cosine_similarity_2d(self, A, B):
        # [T_i, N], [T_i, N]
        A_norm = F.normalize(A, p=2, dim=0) # 对于第j列，即A[:, j]除以了第j列的范数：A[:, j]/sqrt(A[0,j]^2+...+A[T_i-1,j]^2)
        B_norm = F.normalize(B, p=2, dim=0)
        sim_matrix = torch.matmul(A_norm.permute(1, 0), B_norm) # [N, N]
        diagonal_sum = torch.sum(torch.diagonal(sim_matrix)) # \sum_{n}sim(x_1^{b,n}, x_2^{b,n}) 要对于相同n去求余弦相似度
        return diagonal_sum # a value

    def info_loss(self, x_1, x_2, scale_list_proj, i, b, temperature):
        # [T_i, N], [T_i, N], [multiscale, B, T_i, N](这个的T_i是固定的)
        exp_sim = torch.exp(self.cosine_similarity_2d(x_1, x_2) / temperature)
        exp_sum = 0
        for m in range(len(scale_list_proj)):
            if m != i:
                exp_sum_tmp = torch.exp(self.cosine_similarity_2d(x_1, scale_list_proj[m][b, :, :]) / temperature)
                exp_sum += exp_sum_tmp
        return -torch.log(exp_sim / exp_sum)

    def calc_L_constrastive(self, scale_list, temperature=0.2):
        # [multiscale, B, T_i, N], i=1,2,...,i是变化的
        multiscale = len(scale_list)
        B = scale_list[0].shape[0]
        
        L_cc = 0
        for i in range(multiscale - 1):
            scale_list_proj = []
            for m in range(multiscale):
                scale_list_proj.append(self.project_head_layers_ij[i][m](scale_list[m].permute(0, 2, 1)).permute(0, 2, 1)) # T_m->T_i
            # scale_list_proj:[multiscale, B, T_i, N],这里的T_i是固定的，对应于for循环里的i
            for b in range(B):
                x_i = scale_list_proj[i][b, :, :] # [B, T_i, N]
                x_i1 = scale_list_proj[i+1][b, :, :] # [B, T_i, N]
                l_info_1 = self.info_loss(x_i, x_i1, scale_list_proj, i, b, temperature)
                l_info_2 = self.info_loss(x_i1, x_i, scale_list_proj, i+1, b, temperature)
                L_cc += l_info_1 + l_info_2
        
        L_cc /= (2 * B * (multiscale - 1))
        return L_cc