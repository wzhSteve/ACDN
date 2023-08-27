import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import separate_encoder, separate_encoder_layer
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, AttentionLayer
from models.embed import DataEmbedding
from models.simple_linear import simple_linear
from models.encoder import series_decomp
from fft_trans import fft_ifft_picture
from models.distribution_block import distribution_block
from models.fft_decompose import fft_decompose
from models.fft_plus import fft_plus
from draw_picture import draw_picture, draw_trend_res

class separateformer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8,
                 dropout=0.05, attn='full', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, mix=True, separate_factor=2, step=4,
                 device=torch.device('cuda:0')):
        super(separateformer,self).__init__()
        self.seq_len = seq_len
        self.pred_len = out_len #预测序列长度
        self.label_len = label_len
        self.attn = attn #attn模块选取
        self.output_attention = output_attention
        self.separate_factor = separate_factor
        self.dropout = dropout
        self.step = step
        #self.activation = F.gelu if activation == 'gelu' else F.relu
        self.activation = F.elu
        self.d_model = d_model
        self.c_out = c_out

        #encoding ETT中enc_in dec_in都为7 d_model为512，即把七个多元变量通过线性映射到512维上
        self.enc_embedding = DataEmbedding(enc_in,d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in,d_model,embed,freq,dropout)
        #true encoder 2layer
        self.encoder = separate_encoder(self.step, separate_factor=separate_factor,n_heads=n_heads, mix=mix, dropout=dropout, activation=activation, d_model=d_model)
        self.encoder2 = separate_encoder(self.step, separate_factor=separate_factor,n_heads=n_heads, mix=mix, dropout=dropout, activation=activation, d_model=d_model)
        #pred encoder  1layer
        self.encoder_pred = separate_encoder(self.step, separate_factor=separate_factor,n_heads=n_heads, mix=mix, dropout=dropout, activation=activation, d_model=d_model)
        #pred decoder  1layer
        self.decoder = Decoder(self.seq_len, self.label_len, self.pred_len, self.step, self.separate_factor, n_heads, mix,
                               self.dropout, self.d_model, self.c_out, self.activation)
        # linear extract periodical time series
        self.trade_off1 = nn.Parameter(torch.zeros(1, 1, c_out))
        self.trade_off2 = nn.Parameter(torch.ones(1, 1, c_out))

        self.simple_layer = simple_linear(input_dim=self.seq_len, output_dim=self.pred_len + self.label_len)
        self.decompose = series_decomp(seq_len * 3 // 4 + 1) # seq_len * 3 // 4 + 1

        self.dis_block = True
        if self.dis_block:
            self.distribution_block = distribution_block(seq_len=self.seq_len, label_len=self.label_len, pred_len=self.pred_len, feature_dim=c_out, window_size=12)
            self.distribution_weight = nn.Parameter(torch.ones(1, 1, c_out))
            self.distribution_bias = nn.Parameter(torch.zeros(1, 1, c_out))

        self.RIN = True
        if self.RIN:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, c_out))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, c_out))
            self.affine_weight2 = nn.Parameter(torch.ones(1, 1, c_out))
            self.affine_bias2 = nn.Parameter(torch.zeros(1, 1, c_out))

        self.fft_plus = fft_plus(self.pred_len + self.label_len)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,enc_self_mask=None):
        """
        :param x_enc: encoder的输入[batch_size, sequence_len, c_in=7]
        :param x_mark_enc: 输入的时间戳[batche_size, sequence_len, 4]
        :param enc_self_mask:
        :return:
        """

        # dec_out_mean[batch_size, 1, d_model]
        dec_out_mean = torch.mean(x_dec[:, :self.label_len, :], dim=1).view(x_dec.shape[0], 1, x_dec.shape[2])
        # # temp[batch_size, pred_len, d_model]
        temp = dec_out_mean.repeat(1, self.pred_len, 1)
        # temp = torch.ones([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).to(x_dec.device)
        # # dec_out中占位符为已知部分的均值或0 作为预测部分的encoder的输入
        x_dec = x_dec[:, :self.label_len, :]
        # x_dec = x_enc[:, -self.label_len:, :]
        x_dec = torch.cat([x_dec, temp], dim=1)
        x_enc_res, x_enc_trend = self.decompose(x_enc)
        # x_enc_trend, x_enc_res = fft_decompose(x_enc)
        # x_enc_trend = x_enc
        output_linear = self.simple_layer(x_enc_res)
        # draw_picture(x_enc_trend, x_enc_trend, 'decompose trend', 'decompose')
        # draw_picture(x_enc_res, x_enc_res, 'decompose res', 'decompose')

        if self.RIN:
            print('/// RIN ACTIVATED ///\r', end='')
            means1 = x_enc_trend.mean(1, keepdim=True).detach()
            #mean
            x_enc_trend = x_enc_trend - means1
            #var
            stdev1 = torch.sqrt(torch.var(x_enc_trend, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc_trend /= stdev1
            # affine
            # print(x.shape,self.affine_weight.shape,self.affine_bias.shape)
            x_enc_trend = x_enc_trend * self.affine_weight2 + self.affine_bias2

            means = x_dec.mean(1, keepdim=True).detach()
            # mean
            x_dec = x_dec - means
            # var
            stdev = torch.sqrt(torch.var(x_dec, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_dec /= stdev
            # affine
            # print(x.shape,self.affine_weight.shape,self.affine_bias.shape)
            x_dec = x_dec * self.affine_weight + self.affine_bias

        if self.dis_block:
            x_enc_trend, x_dec, label_pred_mean, lable_pred_std, pred_mean, pred_std = self.distribution_block(x_enc_trend, x_dec)
            x_dec = x_dec * self.distribution_weight + self.distribution_bias

        # first embedding
        enc_out = self.enc_embedding(x_enc_trend, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)

        enc_out, layer_output_true = self.encoder(enc_out, attn_mask=enc_self_mask)
        #pred encoder
        dec_out, layer_output_pred = self.encoder_pred(dec_out, attn_mask=enc_self_mask)
        #decoder
        output = self.decoder(enc_out, dec_out, layer_output_true, layer_output_pred)

        if self.dis_block:
            output = output - self.distribution_bias
            output = output / (self.distribution_weight + 1e-10)
            output = output * lable_pred_std
            output = output + label_pred_mean

        ### reverse RIN ###
        if self.RIN:
            output = output - self.affine_bias
            output = output / (self.affine_weight + 1e-10)
            output = output * stdev
            output = output + means

        # draw_picture(output[:, self.label_len:, :], output[:, self.label_len:, :], 'decompose trend pred', 'decompose')
        # draw_picture(output_linear[:, self.label_len:, :], output_linear[:, self.label_len:, :], 'decompose res pred', 'decompose')
        output_final = output + self.trade_off2 * output_linear
        # output = output + self.trade_off1 * self.fft_plus(output)

        return output_final, pred_mean, pred_std, label_pred_mean[:, -self.pred_len:, :], output, output_linear # [B, L, D]




