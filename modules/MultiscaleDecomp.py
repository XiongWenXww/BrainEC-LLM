import torch
import torch.nn as nn
from layers.Autoformer_EncDec import series_decomp
from modules.GroupFFN import GroupFFN


class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, 5)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, args):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        args.ts_len // (args.down_sampling_window ** i),
                        args.ts_len // (args.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        args.ts_len // (args.down_sampling_window ** (i + 1)),
                        args.ts_len // (args.down_sampling_window ** (i + 1)),
                    ),

                )
                for i in range(args.down_sampling_layers)
            ]
        )
        self.groupFFN_layers = torch.nn.ModuleList(
            [
                GroupFFN(
                        args.d_model, args.patch_len, args.stride, args.kernel_size, args.d_ff, args.nodes_num, 
                        args.ts_len, 
                        T_in=args.ts_len // (args.down_sampling_window ** i), # T_i
                        T_out=args.ts_len // (args.down_sampling_window ** (i + 1)), # T_{i+1}
                        drop=args.dropout
                    )
                    for i in range(args.down_sampling_layers)
            ]
        )
        

    def forward(self, season_list):
        # [multiscale, B, T_i, N], T_0 = T, T_1 = T/down_sampling_window, T_2 = T/(down_sampling_window*2), ...
        # mixing high->low
        out_high = season_list[0] # [B, T_i, N]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)] # [1, B, N, T]

        for i in range(len(season_list) - 1):
            out_low_res = self.groupFFN_layers[i](out_high) # [B, N, T_i]->[B, N, T_{i+1}]
            # out_low_res = self.down_sampling_layers[i](out_high) # [B, N, T_i]->[B, N, T_{i+1}]
            out_low = out_low + out_low_res # features of the original layer i+1 + features obtained through the nonlinear change of the layer i
            out_high = out_low
            # i = L-3, out_low = season_list[L-1], at the end of the next cycle, when i=L-2, 
            # out_low is no longer updated, and the loop is just over
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list # [multiscale, B, T_i, N], T_0 = T, T_1 = T/down_sampling_window, T_2 = T/(down_sampling_window*2), ...

class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, args):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        args.ts_len // (args.down_sampling_window ** (i + 1)),
                        args.ts_len // (args.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        args.ts_len // (args.down_sampling_window ** i),
                        args.ts_len // (args.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(args.down_sampling_layers))
            ])
        self.groupFFN_layers = torch.nn.ModuleList(
            [
                GroupFFN(
                        args.d_model, args.patch_len, args.stride, args.kernel_size, args.d_ff, args.nodes_num, 
                        args.ts_len, 
                        T_in=args.ts_len // (args.down_sampling_window ** (i + 1)), # T_{i+1}
                        T_out=args.ts_len // (args.down_sampling_window ** i), # T_i
                        drop=args.dropout
                    )
                    for i in range(args.down_sampling_layers)
            ]
        )

    def forward(self, trend_list):
        # [multiscale, B, T_i, N], T_0 = T, T_1 = T/down_sampling_window, T_2 = T/(down_sampling_window*2), ...
        # mixing low->high
        trend_list_reverse = trend_list.copy()
        # [multiscale, B, T_i, N], T_0 = T/(down_sampling_window*(multiscale-1)), 
        # T_1 = T/(down_sampling_window*(multiscale-2)),... , T_{multiscale-1} = T
        trend_list_reverse.reverse() 
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.groupFFN_layers[len(trend_list_reverse) - 2- i](out_low) # [B, N, T_{i+1}]->[B, N, T_i]
            # out_high_res = self.up_sampling_layers[i](out_low) # [B, N, T_{i+1}]->[B, N, T_i]
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse() # [multiscale, B, T_i, N], T_0 = T, T_1 = T/down_sampling_window, T_2 = T/(down_sampling_window*2), ...
        return out_trend_list # [multiscale, B, T_i, N], T_0 = T, T_1 = T/down_sampling_window, T_2 = T/(down_sampling_window*2), ...


class PastDecomposableMixing(nn.Module):
    def __init__(self, args):
        super(PastDecomposableMixing, self).__init__()
        self.down_sampling_window = args.down_sampling_window

        self.dropout = nn.Dropout(args.dropout)
        self.channel_independence = args.channel_independence

        if args.decomp_method == 'moving_avg':
            self.decomposition = series_decomp(args.moving_avg)
        elif args.decomp_method == "dft_decomp":
            self.decomposition = DFT_series_decomp(args.top_k)
        else:
            raise ValueError('decompsition is error')

        if args.channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=args.nodes_num, out_features=args.d_ff),
                nn.GELU(),
                nn.Linear(in_features=args.d_ff, out_features=args.nodes_num),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(args)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(args)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=args.nodes_num, out_features=args.d_ff),
            nn.GELU(),
            nn.Linear(in_features=args.d_ff, out_features=args.nodes_num),
        )

    def forward(self, x_list):
        # [multiscale, B, T_i, N]
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decomposition(x) # [B, T_i, N], [B, T_i, N], corresponds to fine scales and coarse scales, respectively 
            if self.channel_independence == 0:
                season = self.cross_layer(season) # [B, T_i, N]->[B, T_i, d_ff]->[B, T_i, N]
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))
        # season_list/trend_list: [multiscale, B, T_i, N]

        # Bottom-up season mixing [0->1->...->multiscale]
        out_season_list = self.mixing_multi_scale_season(season_list) # [multiscale, B, T_i, N], T_0 = T, T_1 = T/down_sampling_window, T_2 = T/(down_sampling_window*2), ...
        # top-down trend mixing [multiscale->multiscale-1->...->0]
        out_trend_list = self.mixing_multi_scale_trend(trend_list) # [multiscale, B, T_i, N], T_0 = T, T_1 = T/down_sampling_window, T_2 = T/(down_sampling_window*2), ...

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list, length_list):
            out = out_season + out_trend #[B, T_i, N]
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out)
        return out_list # [multiscale, B, T_i, N]


class MultiscaleDecomp(nn.Module):
    def __init__(self, args):
        super(MultiscaleDecomp, self).__init__()
        self.args = args
        self.ts_len = args.ts_len
        self.down_sampling_window = args.down_sampling_window
        self.channel_independence = args.channel_independence
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(args)
                                         for _ in range(args.e_layers)])

        self.e_layers = args.e_layers

    def __multi_scale_process_inputs(self, x_enc):
        # downsampling to get multsacle time series, return a list of multi-scale input time series
        if self.args.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.args.down_sampling_window, return_indices=False)
        elif self.args.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.args.down_sampling_window)
        elif self.args.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.args.nodes_num, out_channels=self.args.nodes_num,
                                  kernel_size=3, padding=padding,
                                  stride=self.args.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False) # in_channels: the second to last dimension
        else:
            return x_enc
        # B,T,N -> B,N,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        
        x_enc_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1)) # append([B, T, N])
        
        for i in range(self.args.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori) # [B, N, T_{i-1}]->[B, N, T_{i-1}/args.down_sampling_window]

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1)) # append([B, T_i, N])
            x_enc_ori = x_enc_sampling

        x_enc_list = x_enc_sampling_list

        return x_enc_list # [multiscale, B, T_i, N]
    
    def forward(self, x_enc):
        # [B, T, N]
        # downsampling to get multisacle time series
        x_enc_list = self.__multi_scale_process_inputs(x_enc) # [multiscale, B, T_i, N]

        # Past Decomposable Mixing as encoder for past
        for i in range(self.e_layers):
            x_enc_list = self.pdm_blocks[i](x_enc_list) # [multiscale, B, T_i, N]

        return x_enc_list