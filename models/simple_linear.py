import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import series_decomp
import math
import numpy as np



class simple_linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(simple_linear, self).__init__()

        self.hidden_dim = (input_dim + output_dim) // 2
        self.linear = nn.Linear(input_dim, self.hidden_dim, bias=False)
        self.linear_out = nn.Linear(self.hidden_dim, output_dim, bias=False)

    def forward(self, input):
        output = self.linear(input.permute(0, 2, 1))
        output = self.linear_out(output).permute(0, 2, 1)
        return output