import torch
import torch.nn as nn
from .conv_layer import ConvLayer
class DenseSkipBlock(nn.Module):

    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(ConvLayer(in_channels + i * growth_rate, growth_rate))

    def forward(self, x_list):
        out_list = []
        for i, layer in enumerate(self.layers):
            x = torch.cat(x_list, dim=1) 
            x = layer(x)
            x_list.append(x)
            out_list.append(x)
        return out_list[-1]
