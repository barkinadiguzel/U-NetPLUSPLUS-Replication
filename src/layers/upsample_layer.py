import torch
import torch.nn as nn
import torch.nn.functional as F

class UpSampleLayer(nn.Module):
    def __init__(self, scale_factor=2, mode='bilinear', in_channels=None, out_channels=None):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        if mode == 'transpose':
            self.trans_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        if self.mode == 'bilinear':
            return F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        elif self.mode == 'transpose':
            return self.trans_conv(x)
