import torch
import torch.nn as nn
from ..layers.conv_layer import ConvLayer
from ..layers.pool_layers.maxpool_layer import MaxPoolLayer

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, out_channels)
        self.conv2 = ConvLayer(out_channels, out_channels)
        self.pool = MaxPoolLayer()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x_pooled = self.pool(x)
        return x, x_pooled  
