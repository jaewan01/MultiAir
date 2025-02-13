import torch
from torch import nn
import numpy as np

import torch.nn.functional as F
from modules import ConvSC, Inception

# RevIN: Reversible Instance Normalization (https://github.com/ts-kim/RevIN)
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, default_mean=None, default_std=None):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()
        self.default_mean = default_mean
        self.default_std = default_std

    def forward(self, x, mode:str):
        if mode == 'norm':
            #self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features, dtype=torch.float32))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features, dtype=torch.float32))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        mask = ~torch.isnan(x)
        valid_counts = mask.sum(dim=dim2reduce, keepdim=True)
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        sum_squared_diff = torch.where(mask, (x - self.mean)**2, torch.zeros_like(x)).sum(dim=dim2reduce, keepdim=True)
        variance = sum_squared_diff / valid_counts
        
        self.stdev = torch.sqrt(variance + self.eps).detach()
        self.mean = torch.where(torch.isnan(self.mean), torch.FloatTensor([self.default_mean]).cuda(), self.mean)
        self.stdev = torch.where(torch.isnan(self.stdev), torch.FloatTensor([self.default_std]).cuda(), self.stdev)
        self.stdev = torch.where(self.stdev==0, torch.FloatTensor([self.default_std]).cuda(), self.stdev)
        # self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

# positional encoding for raw date and coordinates
class TimeEncode(torch.nn.Module):
    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = torch.nn.Linear(1, dim)
        alpha = int(dim ** (1/2))
        self.w.weight = torch.nn.Parameter((torch.FloatTensor(1 / alpha ** np.linspace(0, alpha-1, dim))).reshape(dim, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, t):
        sin_output = torch.sin(self.w(t.reshape((-1, 1))))
        cos_output = torch.cos(self.w(t.reshape((-1, 1))))
        output = torch.cat([sin_output, cos_output], dim=1)
        return output    
        
# DishTS: Distributional Shift in Time Series (https://github.com/weifantt/Dish-TS)
class DishTS(nn.Module):
    def __init__(self, stn_num, prev_len):
        super().__init__()
        init = 'standard' #'standard', 'avg' or 'uniform'
        activate = True
        n_series = stn_num
        lookback = prev_len

        if init == 'standard':
            self.reduce_mlayer = nn.Parameter(torch.ones(n_series, lookback, 2)/lookback)
        elif init == 'avg':
            self.reduce_mlayer = nn.Parameter(torch.ones(n_series, lookback, 2)/lookback)
        elif init == 'uniform':
            self.reduce_mlayer = nn.Parameter(torch.ones(n_series, lookback, 2)/lookback+torch.rand(n_series, lookback, 2)/lookback)
        self.gamma, self.beta = nn.Parameter(torch.ones(n_series)), nn.Parameter(torch.zeros(n_series))
        self.activate = activate

    def forward(self, batch_x, prev, mode='norm', dec_inp=None):
        if mode == 'norm':
            # batch_x: B*L*D || dec_inp: B*?*D (for xxformers)
            self.preget(batch_x)
            batch_x = self.forward_process(batch_x)
            dec_inp = None if dec_inp is None else self.forward_process(dec_inp)
            return batch_x
        elif mode == 'denorm':
            # batch_x: B*H*D (forecasts)
            batch_y = self.inverse_process(batch_x)
            return batch_y

    def preget(self, batch_x):
        x_transpose = batch_x.permute(2,0,1) 
        theta = torch.bmm(x_transpose, self.reduce_mlayer).permute(1,2,0)
        if self.activate:
            theta = F.gelu(theta)
        self.phil, self.phih = theta[:,:1,:], theta[:,1:,:] 
        self.xil = torch.sum(torch.pow(batch_x - self.phil,2), axis=1, keepdim=True) / (batch_x.shape[1]-1)
        self.xih = torch.sum(torch.pow(batch_x - self.phih,2), axis=1, keepdim=True) / (batch_x.shape[1]-1)
        
        
    def forward_process(self, batch_input):
        temp = (batch_input - self.phil)/torch.sqrt(self.xil + 1e-8)
        rst = temp.mul(self.gamma) + self.beta
        return rst
    
    def inverse_process(self, batch_input):
        return ((batch_input - self.beta) / (self.gamma)) * torch.sqrt(self.xih + 1e-8) + self.phih

def stride_generator(N, reverse=False):
    strides = [1, 2]*10
    if reverse: return list(reversed(strides[:N]))
    else: return strides[:N]


#Encoder for SimVP
class Encoder(nn.Module):
    def __init__(self,C_in, C_hid, N_S):
        super(Encoder,self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )
    
    def forward(self,x):# B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1,len(self.enc)):
            latent = self.enc[i](latent)
        return latent,enc1

#Decoder for SimVP
class Decoder(nn.Module):
    def __init__(self,C_hid, C_out, N_S):
        super(Decoder,self).__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2*C_hid, C_hid, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)
    
    def forward(self, hid, enc1=None):
        for i in range(0,len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        Y = self.readout(Y)
        return Y

# Inception module for SimVP
class Mid_Xnet(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, incep_ker = [3,5,7,11], groups=8):
        super(Mid_Xnet, self).__init__()

        self.N_T = N_T
        enc_layers = [Inception(channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))

        dec_layers = [Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_in, incep_ker= incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)

        # decoder
        z = self.dec[0](z)
        for i in range(1, self.N_T):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z.reshape(B, T, C, H, W)
        return y

# SimVP: Simpler yet Better Video Predicton (https://github.com/A4Bio/SimVP)
class SimVP_adv(nn.Module):
    def __init__(self, shape_in, hid_S=16, hid_T=256, N_S=4, N_T=8, incep_ker=[3,5,7,11], groups=8, C_out=2):
        super(SimVP_adv, self).__init__()
        T, C, H, W = shape_in
        self.enc = Encoder(C, hid_S, N_S)
        self.hid = Mid_Xnet(T*hid_S, hid_T, N_T, incep_ker, groups)
        self.dec = Decoder(hid_S, C, N_S)

    def forward(self, x_raw):

        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B*T, C_, H_, W_)

        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, C, H, W)
        return Y

class MultiAir(nn.Module):
    def __init__(self, input_dim=7, lats = None, lons = None, feat_dim=12, hidden_dim=128, pm25_mean=0, pm25_std=0, output_dim = 6, prev_len=100, korea_stn_num=0, china_stn_num=0, normalization_method = "DishTS"):
        """
        MultiAir: Multi-modal model based on time series and satellite images for Air pollution nowcasting in extreme cases

        :param input_dim: length of time series input
        :param lats,lons: latitude and longitude of stations
        :param feat_dim: number of features
        :param hidden_dim: hidden dimension of LSTM
        :param pm25_mean,pm25_std: mean and std of PM2.5
        :param output_dim: length of time series output
        :param prev_len: length of previous values for normalization
        :param korea_stn_num,china_stn_num: number of stations in Korea and China
        :param normalization_method: normalization method (RevIN, DishTS, Standard)
        """
        super(MultiAir, self).__init__()
        self.input_dim = input_dim 
        self.feat_dim = feat_dim 
        self.hidden_dim = hidden_dim
        self.pm25_mean = pm25_mean
        self.pm25_std = pm25_std
        self.output_dim = output_dim
        self.korea_stn_num = korea_stn_num
        self.china_stn_num = china_stn_num
        self.total_stn_num = korea_stn_num + china_stn_num
        self.prev_len = prev_len
        self.normalization_method = normalization_method

        self.lats = torch.FloatTensor(lats) 
        self.lons = torch.FloatTensor(lons)

        # normalization layer initialization
        if self.normalization_method == "RevIN":
            self.revin_layer = RevIN(self.total_stn_num, default_mean=self.pm25_mean, default_std=self.pm25_std)
        if self.normalization_method == "DishTS":
            self.dishts_layer = DishTS(self.total_stn_num, self.prev_len)
        
        # positional encoding
        self.lat_encoder = TimeEncode(dim=self.hidden_dim//32)
        self.lon_encoder = TimeEncode(dim=self.hidden_dim//32)
        self.month_encoder = TimeEncode(dim=self.hidden_dim//32)
        self.day_encoder = TimeEncode(dim=self.hidden_dim//32)
        self.hour_encoder = TimeEncode(dim=self.hidden_dim//32)

        self.lstmcell = nn.LSTMCell(self.feat_dim + self.hidden_dim // 16 * 5, self.hidden_dim)
        self.decoder = nn.LSTMCell(16, self.hidden_dim)

        self.mha = nn.MultiheadAttention(self.hidden_dim, 1)

        self.last_fc = nn.Linear(self.hidden_dim, 1)   

        self.hidden_init = nn.Parameter(torch.zeros(self.total_stn_num, self.hidden_dim))
        self.cell_init = nn.Parameter(torch.zeros(self.total_stn_num, self.hidden_dim))

        self.last_relu = nn.ReLU()

    def forward(self, feats, masks, raw_times, prev_vals, sat_outputs, sat_inputs):
        batch_size = feats.shape[0]

        lat_feat = self.lat_encoder(self.lats.cuda()).cuda()
        lon_feat = self.lon_encoder(self.lons.cuda()).cuda()

        loc_feats = torch.cat([lat_feat, lon_feat], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1).reshape(batch_size * self.total_stn_num, -1).cuda()

        month_feat = self.month_encoder(raw_times[:,:,0])
        day_feat = self.day_encoder(raw_times[:,:,1])
        hour_feat = self.hour_encoder(raw_times[:,:,2])

        month_feat = month_feat.view(batch_size, self.input_dim + self.output_dim, self.hidden_dim//16).cuda()
        day_feat = day_feat.view(batch_size, self.input_dim + self.output_dim, self.hidden_dim//16).cuda()
        hour_feat = hour_feat.view(batch_size, self.input_dim + self.output_dim, self.hidden_dim//16).cuda()

        time_feat = torch.cat((month_feat, day_feat, hour_feat), dim=-1)
        time_feat = time_feat.permute(1,0,2).unsqueeze(2).repeat(1, 1, self.total_stn_num, 1).view(self.input_dim + self.output_dim, batch_size * self.total_stn_num, -1).cuda()

        feats = feats.permute(1, 0, 2, 3) # (#inputdim, #batch, #stn, #featnum) 
        
        # normalization
        if self.normalization_method == "RevIN":
            self.revin_layer._get_statistics(prev_vals)
            norm_PM = self.revin_layer(feats[:,:,:,0].permute(1,0,2), 'norm')
        if self.normalization_method == "DishTS":
            norm_PM = self.dishts_layer(feats[:,:,:,0].permute(1,0,2), prev_vals, 'norm')
        if self.normalization_method == "Standard":
            norm_PM = (feats[:,:,:,0].permute(1,0,2) - self.pm25_mean) / self.pm25_std

        feats = feats.clone()
        feats[:,:,:,0] = norm_PM.permute(1,0,2)

        feats = feats.contiguous().cuda()

        curr_hidden_state = self.hidden_init.unsqueeze(0).repeat(batch_size, 1, 1).cuda()
        curr_cell_state = self.cell_init.unsqueeze(0).repeat(batch_size, 1, 1).view(-1, self.hidden_dim).cuda()

        for i in range(self.input_dim):

            curr_input = torch.cat([feats[i].view(batch_size * self.total_stn_num, -1), time_feat[i], loc_feats], dim=-1)

            curr_hidden_state, curr_cell_state = self.lstmcell(curr_input, (curr_hidden_state.view(-1, self.hidden_dim), curr_cell_state)) # ((#batch * #stn), #hdim)
            curr_hidden_state = curr_hidden_state.view(batch_size, self.total_stn_num, self.hidden_dim)

            curr_attn = self.mha(curr_hidden_state.permute(1,0,2), curr_hidden_state.permute(1,0,2), curr_hidden_state.permute(1,0,2), key_padding_mask = ~masks)[0]
            curr_hidden_state = curr_hidden_state + curr_attn.permute(1,0,2)

        sat_outputs_mean = sat_outputs.mean(dim=1)
        sat_outputs_mean = sat_outputs_mean.unsqueeze(1).repeat(1, self.total_stn_num, 1).view(batch_size * self.total_stn_num, -1)
        sat_outputs_std = sat_outputs.std(dim=1)
        sat_outputs_std = sat_outputs_std.unsqueeze(1).repeat(1, self.total_stn_num, 1).view(batch_size * self.total_stn_num, -1)

        sat_outputs = sat_outputs.view(batch_size * self.total_stn_num, -1)
        sat_inputs = sat_inputs.view(batch_size * self.total_stn_num, -1)
        sat_inputs[sat_inputs == -1] = 0

        preds= []

        for i in range(self.output_dim):
            
            # decoder input consisting of (a) previous satellite images, (b) satellite predictions, (c) mean, and (d) std of satellite predictions
            curr_input = torch.cat([sat_inputs, sat_outputs[:,i].unsqueeze(-1),sat_outputs_mean[:,i].unsqueeze(-1), sat_outputs_std[:,i].unsqueeze(-1)], dim=-1)
            curr_hidden_state, curr_cell_state = self.decoder(curr_input, (curr_hidden_state.view(-1, self.hidden_dim), curr_cell_state)) # ((#batch * #stn), #hdim)
            curr_hidden_state = curr_hidden_state.view(batch_size, self.total_stn_num, self.hidden_dim)

            curr_attn = self.mha(curr_hidden_state.permute(1,0,2), curr_hidden_state.permute(1,0,2), curr_hidden_state.permute(1,0,2), key_padding_mask = ~masks)[0]
            curr_hidden_state = curr_hidden_state + curr_attn.permute(1,0,2)

            result = self.last_fc(curr_hidden_state).contiguous()

            if self.normalization_method == "RevIN":
                pred = self.revin_layer(result.permute(0,2,1), 'denorm')[:,:,:self.korea_stn_num].permute(0,2,1)
            if self.normalization_method == "DishTS":
                pred = self.dishts_layer(result.permute(0,2,1), prev_vals, 'denorm')[:,:,:self.korea_stn_num].permute(0,2,1)
            if self.normalization_method == "Standard":
                pred = result[:,:self.korea_stn_num]

            preds.append(self.last_relu(pred))

        preds = torch.cat(preds, dim=-1)

        return preds
