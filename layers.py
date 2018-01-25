import torch.nn as nn


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=6, kernel_size=3, stride=2):
        super(InvertedResidual, self).__init__()

        if stride != 1 and stride != 2:
            raise ValueError("Stride should be 1 or 2")

        self.block = nn.Sequential(nn.Conv2d(in_channels, in_channels * expansion_factor, 1, bias=False),
                                   nn.BatchNorm2d(in_channels * expansion_factor),
                                   nn.ReLU6(inplace=True),

                                   nn.Conv2d(in_channels * expansion_factor, in_channels * expansion_factor,
                                             kernel_size, stride, 1,
                                             groups=in_channels * expansion_factor, bias=False),
                                   nn.BatchNorm2d(in_channels * expansion_factor),
                                   nn.ReLU6(inplace=True),

                                   nn.Conv2d(in_channels * expansion_factor, out_channels, 1,
                                             bias=False),
                                   nn.BatchNorm2d(out_channels))

        self.is_residual = False if stride == 2 else True

    def forward(self, x):
        block = self.block(x)
        if self.is_residual:
            return x + block
        return block


def bottleneck_sequence(in_channels, out_channels, num_units, expansion_factor=6, kernel_size=3, initial_stride=2):
    bottleneck_arr = [InvertedResidual(in_channels, out_channels, expansion_factor, kernel_size, initial_stride)]

    for i in range(num_units - 1):
        bottleneck_arr.append(InvertedResidual(out_channels, out_channels, expansion_factor, kernel_size, 1))

    return nn.Sequential(*bottleneck_arr)


def conv2d_bn_relu(in_channels, out_channels, kernel_size=3, stride=2, dropout_prob=0.0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        # For efficiency, dropout is placed before relu.
        nn.Dropout2d(dropout_prob, inplace=True),
        nn.ReLU6(inplace=True)
    )
