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

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from math import sqrt
import os

def fft_decompose(input):
    input_fft = torch.fft.rfft(input, dim=1)
    temp = torch.zeros([input.shape[0], input_fft.shape[1], input.shape[2]]).to(input.device)
    trend = torch.cat([input_fft[:, :(input_fft.shape[1]-1)//4+1, :], temp[:, -input_fft.shape[1]+((input_fft.shape[1]-1)//4+1):, :]], dim=1).float()
    seasonal = torch.cat([temp[:, :(input_fft.shape[1]-1)//4+1, :], input_fft[:, -input_fft.shape[1]+((input_fft.shape[1]-1)//4+1):, :]], dim=1).float()
    trend = torch.fft.irfft(trend, dim=1)
    seasonal = torch.fft.irfft(seasonal, dim=1)
    return trend, seasonal