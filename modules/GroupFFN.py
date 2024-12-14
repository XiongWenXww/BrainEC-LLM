from torch import nn
from layers.Embed import PatchEmbedding

class GroupFFN(nn.Module):
    def __init__(self, d_model, patch_len, patch_stride, kernel_size, dff, nvars, ts_len, T_in, T_out, drop=0.1):
        super(GroupFFN, self).__init__()
        self.ts_len = ts_len
        self.patch_embedding = PatchEmbedding(d_model, patch_len, patch_stride, drop)
        
        self.dw = nn.Conv1d(in_channels=nvars * d_model, out_channels=nvars * d_model, kernel_size=kernel_size,
                                         stride=1, padding=kernel_size// 2, dilation=1, groups=nvars * d_model, bias=True)
        self.norm = nn.BatchNorm1d(d_model)

        #convffn1
        self.ffn1pw1 = nn.Conv1d(in_channels=nvars * d_model, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * d_model, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        #convffn2
        self.ffn2pw1 = nn.Conv1d(in_channels=nvars * d_model, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=d_model)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * d_model, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=d_model)
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

        self.ffn_ratio = dff//d_model

        # self.inverse_value_embedding = nn.Conv1d(in_channels=d_model, out_channels=patch_len, kernel_size=1)
        self.inverse_value_embedding = nn.Linear(in_features=d_model, out_features=patch_len)
        self.L = int((T_in+patch_stride-patch_len)/patch_stride + 1)
        self.projection = nn.Conv1d(in_channels=self.L*patch_len, out_channels=T_out, kernel_size=1)

    def out_projection(self, x):
        # [B, N, d_model, L]
        out = self.inverse_value_embedding(x.permute(0, 1, 3, 2)) # [B, N, L, patch_len]
        out = out.view(out.shape[0], out.shape[1], -1) # [B, N, L*patch_len]
        out = self.projection(out.permute(0, 2, 1)).permute(0, 2, 1) # [B, N, T_out]
        return out


    def forward(self,x):
        # [B, N, T_in]
        B, N, T_in = x.shape
        x, _ = self.patch_embedding(x) # [B*N, (T_in+stride-patch_len)/stride+1, d_model]
        # ****** def L = (T_in+stride-patch_len)/stride+1
        x = x.view(B, N, x.shape[1], x.shape[2]).permute(0, 1, 3, 2) # [B, N, d_model, L]
        input = x
        B, N, d_model, L = x.shape
        x = x.reshape(B,N*d_model,L)
        x = self.dw(x)
        x = x.reshape(B,N,d_model,L)
        x = x.reshape(B*N,d_model,L)
        x = self.norm(x)
        x = x.reshape(B, N, d_model, L)
        x = x.reshape(B, N * d_model, L)

        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(B, N, d_model, L)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, d_model * N, L)
        x = self.ffn2drop1(self.ffn2pw1(x))
        x = self.ffn2act(x)
        x = self.ffn2drop2(self.ffn2pw2(x))
        x = x.reshape(B, d_model, N, L)
        x = x.permute(0, 2, 1, 3) # [B, N, d_model, L]

        x = input + x
        out = self.out_projection(x) # [B, N, d_model, L]->[B, N, T_out]
        return out
