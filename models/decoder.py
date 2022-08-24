import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attn import FullAttention, AttentionLayer


class separate_decoder_layer(nn.Module):
    """
    功能：
    """
    def __init__(self, d_model, dropout=0.1, activation="gelu", separate_factor=2, step=4):
        super(separate_decoder_layer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv1d(in_channels=step, out_channels=step * separate_factor, kernel_size=5, padding=2, bias=False)
        self.activation = F.elu

    def forward(self, x):
        y = self.dropout(self.activation(self.conv(x)))
        return y


class DecoderLayer(nn.Module):
    def __init__(self, L_in, L_out, dropout=0.1, activation="gelu"):
        super(DecoderLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels=L_in, out_channels=L_out, kernel_size=5, padding=2, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.elu

    def forward(self, x):
        y = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        y = self.dropout(self.activation(y))
        return y


class Decoder(nn.Module):
    def __init__(self, seq_len, label_len, pred_len, step, separate_factor, n_heads, mix, dropout=0.1, d_model=512, c_out=7, activation='gelu'):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len + label_len
        self.dropout = dropout
        self.d_model = d_model
        self.c_out = c_out

        self.separate_factor = [3, 2]  # 第一层，二层，三层
        self.step = [24, 8, 4]
        self.layer_len = 2
        self.activation = F.elu

        self.true_encoder_list = nn.ModuleList([]) #用于转换true encoder和pred encoder之间的维度
        self.conv_module = nn.ModuleList([]) #用于true和pred两个encoder之间层间输出进行合并 输入cat(true, pred) 输出降维
        self.module = nn.ModuleList([])  # 用于decoder逆向进行维度变换（小->大）
        self.module2 = nn.ModuleList([])

        self.linear = nn.Linear(self.pred_len, self.pred_len)

        count = 0
        sequence_len = seq_len
        while(count < self.layer_len):
            old_pred = self.pred_len
            sequence_len = sequence_len//self.separate_factor[count]
            self.pred_len = self.pred_len//self.separate_factor[count]

            self.module.append(separate_decoder_layer(d_model, dropout=dropout,
                                       activation=activation, step=self.step[self.layer_len-count], separate_factor=self.separate_factor[self.layer_len-1-count]))
            self.module2.append(separate_decoder_layer(d_model, dropout=dropout,
                                                      activation=activation, step=self.step[self.layer_len - count],
                                                      separate_factor=self.separate_factor[self.layer_len - 1 - count]))
            self.conv_module.append(DecoderLayer(2*self.pred_len, self.pred_len, self.dropout, self.activation))
            self.true_encoder_list.append(DecoderLayer(sequence_len, self.pred_len, self.dropout, self.activation))
            count = count + 1
        #true encoder to z_1
        self.conv_output = nn.Conv1d(in_channels=self.pred_len, out_channels=self.pred_len, kernel_size=5, padding=2, bias=False)
        self.conv_z1 = nn.Conv1d(in_channels=sequence_len, out_channels=self.pred_len, kernel_size=5, padding=2, bias=False)
        #true pred encoder to z
        self.linear_enc_true_pred = nn.Linear(2 * self.pred_len, self.pred_len)
        #z to decoder
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=5, stride=1, padding=2,
                                    padding_mode='circular', bias=False)
        self.dropout_out = nn.Dropout(dropout)


    def forward(self, enc_out_true, enc_out_pred, layer_output_true, layer_output_pred):
        #z
        enc_out_true = self.conv_z1(enc_out_true).transpose(-1, -2)  # B D L
        output = self.dropout_out(self.activation(self.conv_output(enc_out_pred + enc_out_true.transpose(-1, -2)))).transpose(-1, -2)
        layer_len = self.layer_len
        while(layer_len > 0):
            #下标
            layer_len = layer_len - 1
            #layer_output_temp[batch_size, d_model, L]
            #true encoder维度变换到pred encoder
            layer_output_temp = self.true_encoder_list[layer_len](layer_output_true[layer_len].transpose(-1, -2))#B D L
            #拼接true encoder pred encoder，下一步输入decoder中
            layer_output_temp = self.conv_module[layer_len](torch.cat([layer_output_temp, layer_output_pred[layer_len].transpose(-1, -2)], dim=-1)) #B D L
            #output为全局特征，layer_output_temp为局部特征，对其进行维度变换，与encoder相反
            output = output.transpose(-1, -2)#B L D
            layer_output_temp = layer_output_temp.transpose(-1, -2)#B L D
            cnt = layer_output_temp.shape[1] // self.step[layer_len+1]  # 该层块的个数cnt
            #用于存放本层的输出
            next_output = torch.tensor([]).to(output.device)
            for i in range(cnt):
                ii = i * self.step[layer_len+1]
                output_temp = output[:, ii:ii + self.step[layer_len+1], :]
                layer_output_temp_temp = layer_output_temp[:, ii:ii + self.step[layer_len+1], :]
                # 将通过attention后的全局特征和局部特征相加
                output_temp = self.module[self.layer_len - 1 -layer_len](output_temp)
                output_div_temp = self.module2[self.layer_len-1-layer_len](layer_output_temp_temp)#B L D
                output_temp = output_temp + output_div_temp
                next_output = torch.cat([next_output, output_temp], dim=1)#B L D
            output = next_output.transpose(-1, -2)#B D L
        output = output.transpose(-1, -2)#B L D
        output = self.projection(output.permute(0, 2, 1)).transpose(1, 2)
        output = self.linear(output.permute(0, 2, 1)).permute(0, 2, 1)
        return output