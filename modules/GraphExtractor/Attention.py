import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # [B, h, N, d_k]
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3)) # [B, h, N, N]

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        N = attn.shape[-1]
        attn[:, :, torch.arange(N), torch.arange(N)] = -1e9 # 设置对角线为0
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        # [B, N, T]
        # q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # [B, T, N]->[B, N, T]
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # [B, N, T]->[B, N, h*d_k]->[B, N, h, d_k]
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # [B, h, N, d_k]

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask) #  # [B, h, N, d_k]，[B, h, N, N]

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1) # [B, h, N, d_k]->[B, N, h, d_k]->[B, N, h*d_k]

        q = self.dropout(self.fc(q)) # [B, N, h*d_k]->[B, N, T]
        q += residual

        q = self.layer_norm(q) # [B, N, T]

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return self.pos_table[:, :x.size(1)].clone().detach()

    
class AttEncoderLayer(nn.Module):
    ''' Compose with two layers '''
    # d_model is T while data's dimension is [B, T, N]
    def __init__(self, args, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(AttEncoderLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_enc = PositionalEncoding(d_hid=args.nodes_num, n_position=args.ts_len).to(args.device)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6).to(args.device)
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        # position encoding
        pos_enc = self.position_enc(enc_input) # [1, T, N]
        pos_enc = pos_enc.expand(enc_input.shape) # [1, T, N]->[B, T, N]
        enc_output = self.dropout(enc_input + pos_enc)
        enc_output = enc_output.transpose(1, 2) # [B, N, T]
        enc_output = self.layer_norm(enc_output)

        residual = enc_output
        enc_output, enc_slf_attn = self.slf_attn(
            enc_output, enc_output, enc_output, mask=slf_attn_mask) # [B, N, T], [B, h, N, N]
        enc_output = self.pos_ffn(enc_output)
        
        enc_output += residual
        enc_output = enc_output.transpose(1, 2) # [B, T, N]

        enc_slf_attn = torch.mean(torch.mean(enc_slf_attn, dim=0), dim=0)  # [B, h, N, N]->[N, N], att_ij means j cause causal effect on i
        enc_slf_attn = torch.t(enc_slf_attn) # transpose
        # print('attn each row sum:', torch.sum(enc_slf_attn, dim=-1))
        return enc_output, enc_slf_attn