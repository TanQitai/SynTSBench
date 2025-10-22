import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PGN_2d(nn.Module):
    def __init__(self, seq_R, freq, c_in, c_out, windows_size):
        super(PGN_2d, self).__init__()
        
        self.seq_R = seq_R
        self.freq = freq
        self.c_out = c_out
        self.windows_size = windows_size
        
        if freq == 't':
            dim_time = 5
        elif freq == 'h':
            dim_time = 4
        if freq == 'd':
            dim_time = 3
        
        self.hidden_MLP = nn.Conv1d(
            in_channels = c_in * (1 + dim_time), 
            out_channels = c_in * c_out, 
            kernel_size = windows_size, 
            stride = 1, groups = c_in)

        self.gate = nn.Conv1d(
            in_channels = c_in * (1 + dim_time + c_out), 
            out_channels = c_in * 2 * c_out, 
            kernel_size = 1, 
            stride = 1, groups = c_in)
        
        self.fc = nn.Conv1d(
            in_channels = c_in * c_out, 
            out_channels = c_in * c_out, 
            kernel_size = seq_R, 
            stride = 1, groups = c_in)
    
    def deal(self, x, x_mark):
        B, R, C, c_in, _ = x.shape
        c_time = x_mark.shape[-1]
        x_input = torch.cat([x, x_mark], dim=-1)
        x_supply = torch.zeros(B, self.windows_size, C, 
            c_in, (1 + c_time)).to(x.device)
        x_all = torch.cat([x_supply, x_input], dim=1).permute(0, 2, 1, 3, 4)
        x_all = x_all.reshape(B * C, R + self.windows_size, 
            c_in * (1 + c_time)).permute(0, 2, 1)
        x_all_out = self.hidden_MLP(x_all[:, :, :-1]).reshape(
            B, C, c_in, self.c_out, R).permute(0, 4, 1, 2, 3)
        return x_all_out

    def gated_unit(self, x, x_mark, hid):
        x = torch.cat([x, x_mark, hid], dim=-1)
        B, R, C, c_in, c_all = x.shape
        x = x.reshape(B * R * C, c_in * c_all, 1)
        x_embed = self.gate(x).reshape(B, R, C, c_in, -1)
        sigmod_gate, tanh_gate = torch.split(x_embed, self.c_out, dim = -1)
        sigmod_gate = torch.sigmoid(sigmod_gate)
        tanh_gate = torch.tanh(tanh_gate)
        hid = hid * sigmod_gate + (1 - sigmod_gate) * tanh_gate
        return hid
    
    def forward(self, x, x_mark):
        B, R, C, c_in, _ = x.shape
        c_time = x_mark.shape[-1]
        out = self.deal(x, x_mark)
        out = self.gated_unit(x, x_mark, out)
        out = self.fc(out.permute(0, 2, 3, 4, 1).reshape(
            B * C, c_in * self.c_out, R)).reshape(B, C, c_in, self.c_out)
        return out

class short_term_deal(nn.Module):
    def __init__(self, seq_R, freq, c_in, c_out, period):
        super(short_term_deal, self).__init__()
        
        self.seq_R = seq_R
        self.freq = freq
        self.c_out = c_out
        self.period = period
        
        if freq == 't':
            dim_time = 5
        elif freq == 'h':
            dim_time = 4
        if freq == 'd':
            dim_time = 3
        
        self.fc_row = nn.Conv1d(
            in_channels = c_in * (1 + dim_time), 
            out_channels = c_in * c_out, 
            kernel_size = period, 
            stride = 1, groups = c_in)
        
        self.fc_col = nn.Conv1d(
            in_channels = c_in * c_out, 
            out_channels = c_in * c_out, 
            kernel_size = seq_R, 
            stride = 1, groups = c_in)
    
    def forward(self, x, x_mark):
        B, R, C, c_in, _ = x.shape
        c_time = x_mark.shape[-1]
        x_input = torch.cat([x, x_mark], dim=-1)
        out = self.fc_row(x_input.permute(0, 1, 3, 4, 2).reshape(
            B * R, c_in * (1 + c_time), C)).reshape(B, R, c_in * self.c_out)
        out = self.fc_col(out.permute(0, 2, 1)).reshape(
            B, c_in, 1, self.c_out).repeat(1, 1, self.period, 1)
        return out.permute(0, 2, 1, 3)

class TPGN(nn.Module):
    def __init__(self, seq_R, freq, c_in, c_out, windows_size, 
            period, pred_R, need_short=1):
        super(TPGN, self).__init__()
        
        self.freq = freq
        self.c_in = c_in
        self.c_out = c_out
        self.windows_size = windows_size
        self.pred_R = pred_R
        self.need_short = need_short
        
        self.LNN_dim = PGN_2d(seq_R, freq, c_in, c_out, windows_size)
        
        if self.need_short:
            self.s_t_p_e = short_term_deal(seq_R, 
                freq, c_in, c_out, period)
            
            self.fc = nn.Conv1d(
                in_channels = c_in * 2 * c_out, 
                out_channels = c_in * pred_R, 
                kernel_size = 1, 
                stride = 1, groups = c_in)
        else:
            self.fc = nn.Conv1d(
                in_channels = c_in * c_out, 
                out_channels = c_in * pred_R, 
                kernel_size = 1, 
                stride = 1, groups = c_in)
    
    def forward(self, x, x_mark):
        B, R, C, c_in = x.shape
        c_time = x_mark.shape[-1]
        x = x.unsqueeze(-1)
        x_mark = x_mark.unsqueeze(-2).repeat(1, 1, 1, c_in, 1)
        out_long_term = self.LNN_dim(x, x_mark)
        if self.need_short:
            out_short_term = self.s_t_p_e(x, x_mark)
            out_all = torch.cat([out_short_term, out_long_term], 
                dim=-1).reshape(B * C, c_in * 2 * self.c_out, 1)
        else:
            out_all = out_long_term.reshape(
                B * C, c_in * self.c_out, 1)
            
        out_all = self.fc(out_all).reshape(B, C, c_in, self.pred_R).permute(
            0, 3, 1, 2).reshape(B, -1, c_in)
        return out_all

class Model(nn.Module):
    """
    TPGN Model for Time Series Forecasting
    Paper link: https://arxiv.org/abs/your_paper_link_here
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.configs = configs
        self.task_name = getattr(configs, 'task_name', 'long_term_forecast')

        self.freq = configs.freq
        # 设置默认周期，如果未指定则根据频率设置
        if hasattr(configs, 'TPGN_period'):
            self.period = configs.TPGN_period
        else:
            # 根据频率设置默认周期
            if configs.freq == 'h':
                self.period = 24  # 24小时周期
            elif configs.freq == 'd':
                self.period = 7   # 7天周期
            elif configs.freq == 't':
                self.period = 60  # 60分钟周期
            else:
                self.period = 24  # 默认周期
        
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.seq_R = int(math.ceil(self.seq_len/self.period))
        self.pred_R = int(math.ceil(self.pred_len/self.period))
        
        self.c_in = configs.enc_in
        self.c_out = configs.c_out
        
        self.d_model = configs.d_model
        # 使用use_norm参数，兼容框架
        self.norm = getattr(configs, 'use_norm', 1)
        
        self.TPGN = TPGN(self.seq_R, self.freq, self.c_in, 
            self.d_model, self.seq_R-1, self.period, self.pred_R)
        
        # 为其他任务添加投影层
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)
        
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x: [Batch, Input length, Channel]
        B, L, c_in = x_enc.shape
        c_time = x_mark_enc.shape[-1]
        
        if self.norm == 1:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, 
                keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        
        x_enc = x_enc.reshape(B, self.seq_R, self.period, c_in)
        x_mark_enc = x_mark_enc.reshape(B, self.seq_R, self.period, c_time)
        
        output = self.TPGN(x_enc, x_mark_enc)
        
        if self.norm == 1:
            output = output * (stdev[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len, 1))
            output = output + (means[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len, 1))
        
        return output
    
    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # 对于填补任务，返回输入相同长度的输出
        B, L, c_in = x_enc.shape
        c_time = x_mark_enc.shape[-1]
        
        if self.norm == 1:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, 
                keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        
        # 调整形状以适应TPGN
        seq_R_imp = int(math.ceil(L/self.period))
        x_enc_reshaped = x_enc.reshape(B, seq_R_imp, self.period, c_in)
        x_mark_enc_reshaped = x_mark_enc.reshape(B, seq_R_imp, self.period, c_time)
        
        # 使用TPGN进行特征提取
        tpgn_out = self.TPGN.LNN_dim(x_enc_reshaped.unsqueeze(-1), 
                                     x_mark_enc_reshaped.unsqueeze(-2).repeat(1, 1, 1, c_in, 1))
        
        # 重塑回原始形状
        output = tpgn_out.permute(0, 2, 1, 3).reshape(B, L, c_in)
        
        if self.norm == 1:
            output = output * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
            output = output + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        
        return output
        
    def anomaly_detection(self, x_enc):
        # 对于异常检测任务
        B, L, c_in = x_enc.shape
        
        if self.norm == 1:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, 
                keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        
        # 调整形状以适应TPGN
        seq_R_anom = int(math.ceil(L/self.period))
        x_enc_reshaped = x_enc.reshape(B, seq_R_anom, self.period, c_in)
        
        # 创建虚拟时间标记
        x_mark_enc = torch.zeros((B, L, 4)).to(x_enc.device)  # 假设时间特征维度为4
        x_mark_enc_reshaped = x_mark_enc.reshape(B, seq_R_anom, self.period, 4)
        
        # 使用TPGN进行特征提取
        tpgn_out = self.TPGN.LNN_dim(x_enc_reshaped.unsqueeze(-1), 
                                     x_mark_enc_reshaped.unsqueeze(-2).repeat(1, 1, 1, c_in, 1))
        
        # 重塑回原始形状
        output = tpgn_out.permute(0, 2, 1, 3).reshape(B, L, c_in)
        
        if self.norm == 1:
            output = output * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
            output = output + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        
        return output
        
    def classification(self, x_enc, x_mark_enc):
        # 对于分类任务
        B, L, c_in = x_enc.shape
        c_time = x_mark_enc.shape[-1]
        
        if self.norm == 1:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, 
                keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        
        x_enc = x_enc.reshape(B, self.seq_R, self.period, c_in)
        x_mark_enc = x_mark_enc.reshape(B, self.seq_R, self.period, c_time)
        
        # 使用TPGN进行特征提取
        tpgn_out = self.TPGN.LNN_dim(x_enc.unsqueeze(-1), 
                                     x_mark_enc.unsqueeze(-2).repeat(1, 1, 1, c_in, 1))
        
        # 展平并分类
        enc_out = tpgn_out.reshape(B, -1)
        enc_out = self.act(enc_out)
        enc_out = self.dropout(enc_out)
        output = self.projection(enc_out)
        
        return output
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None