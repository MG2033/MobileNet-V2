import torch.nn as nn


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, kernel_size, stride, groups, bias=False):
        super(InvertedResidual, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, in_channels * expansion_factor, 1, bias=bias),
                                   nn.ReLU6(inplace=True))


    def forward(self, x):
        pass
