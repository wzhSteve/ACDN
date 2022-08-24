import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
from scipy import signal

from models.attn import FullAttention, AttentionLayer


class fft_plus(nn.Module):
    def __init__(self, L):
        super(fft_plus, self).__init__()
        self.conv = nn.Conv1d(in_channels=2*L, out_channels=L, kernel_size=3, padding=1, bias=False)
        self.conv_1 = nn.Conv1d(in_channels=L//2+1, out_channels=L//2+1, kernel_size=3, padding=1, bias=False)
        self.conv_2 = nn.Conv1d(in_channels=L//2+1, out_channels=L//2+1, kernel_size=3, padding=1, bias=False)
        self.linearl = nn.Linear(L//2+1, L//2+1)
        self.linearr = nn.Linear(L//2+1, L//2+1)
        self.dropout = nn.Dropout(0.01)
        self.activation = F.elu

    def forward(self, input):

        input_fft = torch.fft.rfft(input, dim=1)
        input_real = input_fft.real
        input_imag = input_fft.imag
        output_real = self.conv_1(input_real)
        output_imag = self.conv_2(input_imag)
        output = torch.fft.irfft(torch.complex(output_real, output_imag), dim=1)

        return output