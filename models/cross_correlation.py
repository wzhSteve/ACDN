import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
from scipy import signal

from models.attn import FullAttention, AttentionLayer


class cross_correlation(nn.Module):
    def __init__(self, L):
        super(cross_correlation, self).__init__()
        self.conv = nn.Conv1d(in_channels=2*L, out_channels=L, kernel_size=3, padding=1, bias=False)
        self.conv_1 = nn.Conv1d(in_channels=L, out_channels=L, kernel_size=3, padding=1, bias=False)
        self.conv_2 = nn.Conv1d(in_channels=L, out_channels=L, kernel_size=3, padding=1, bias=False)
        self.linearl = nn.Linear(L, L)
        self.linearr = nn.Linear(L, L)
        self.dropout = nn.Dropout(0.01)
        self.activation = F.elu

    def forward(self, left, right):
        left = self.linearl(left.permute(0, 2, 1)).permute(0, 2, 1)
        right = self.linearr(right.permute(0, 2, 1)).permute(0, 2, 1)
        left_padding = torch.zeros_like(left).float()
        left_padding = torch.cat([left, left_padding], dim=1)
        right_padding = torch.zeros_like(right).float()
        right_padding = torch.cat([right_padding, right], dim=1)
        count = 2 * left.shape[1]
        out = torch.tensor([]).to(left.device)
        for i in range(count):
            temp_left = left_padding[:, :i+1, :]
            temp_right = right_padding[:, -i-1:, :]
            temp = temp_left * temp_right
            out = torch.cat([out, torch.sum(temp, dim=1, keepdim=True)], dim=1)
        out = self.conv(out)

        # temp = self.conv_1(left)
        # out = temp + right
        # out = self.dropout(self.activation(self.conv_2(out)))
        return out