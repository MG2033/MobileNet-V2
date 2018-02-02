import torch.nn as nn


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=6, kernel_size=3, stride=2):
        super(InvertedResidualBlock, self).__init__()

        if stride != 1 and stride != 2:
            raise ValueError("Stride should be 1 or 2")

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expansion_factor, 1, bias=False),
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

        self.is_residual = True if stride == 1 else False
        self.is_conv_res = False if in_channels == out_channels else True

        # Assumption based on previous ResNet papers: If the number of filters doesn't match,
        # there should be a conv1x1 operation.
        if stride == 1 and self.is_conv_res:
            self.conv_res = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                          nn.BatchNorm2d(out_channels))

    def forward(self, x):
        block = self.block(x)
        if self.is_residual:
            if self.is_conv_res:
                return self.conv_res(x) + block
            return x + block
        return block


def inverted_residual_sequence(in_channels, out_channels, num_units, expansion_factor=6,
                               kernel_size=3,
                               initial_stride=2):
    bottleneck_arr = [
        InvertedResidualBlock(in_channels, out_channels, expansion_factor, kernel_size,
                              initial_stride)]

    for i in range(num_units - 1):
        bottleneck_arr.append(
            InvertedResidualBlock(out_channels, out_channels, expansion_factor, kernel_size, 1))

    return bottleneck_arr


def conv2d_bn_relu6(in_channels, out_channels, kernel_size=3, stride=2, dropout_prob=0.0):
    # To preserve the equation of padding. (k=1 maps to pad 0, k=3 maps to pad 1, k=5 maps to pad 2, etc.)
    padding = (kernel_size + 1) // 2 - 1
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        # For efficiency, Dropout is placed before Relu.
        nn.Dropout2d(dropout_prob, inplace=True),
        # Assumption: Relu6 is used everywhere.
        nn.ReLU6(inplace=True)
    )
