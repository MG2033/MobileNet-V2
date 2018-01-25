import torch.nn as nn
import torch.nn.functional as F
from layers import bottleneck_sequence, conv2d_bn_relu


class MobileNetV2(nn.Module):
    def __init__(self, args):
        super(MobileNetV2, self).__init__()

        # Network is created here, then will be unpacked into nn.sequential
        self.network_settings = [{'t': None, 'c': 32, 'n': 1, 's': 2},
                                 {'t': 1, 'c': 16, 'n': 1, 's': 1},
                                 {'t': 6, 'c': 24, 'n': 2, 's': 2},
                                 {'t': 6, 'c': 32, 'n': 3, 's': 2},
                                 {'t': 6, 'c': 64, 'n': 4, 's': 2},
                                 {'t': 6, 'c': 96, 'n': 3, 's': 1},
                                 {'t': 6, 'c': 160, 'n': 3, 's': 2},
                                 {'t': 6, 'c': 320, 'n': 1, 's': 1},
                                 {'t': None, 'c': 1280, 'n': 1, 's': 1}]
        self.num_classes = args.num_classes
        self.bias = True if args.bias != -1 else False
        self.bias_init = args.bias

        ###############################################################################################################

        # Feature Extraction part
        # Layer 0
        self.network = [
            conv2d_bn_relu(args.num_channels, int(self.network_settings[0]['c'] * args.width_multiplier),
                           args.kernel_size,
                           1, args.dropout_prob, self.bias)]

        # Layers from 1 to 7
        for i in range(1, 8):
            self.network.append(bottleneck_sequence(int(self.network_settings[i - 1]['c'] * args.width_multiplier),
                                                    int(self.network_settings[i]['c'] * args.width_multiplier),
                                                    self.network_settings[i]['n'], self.network_settings[i]['t'],
                                                    args.kernel_size, self.network_settings[i]['s'], self.bias))

        # Last layer before flattening
        self.network.append(
            conv2d_bn_relu(int(self.network_settings[7]['c'] * args.width_multiplier),
                           int(self.network_settings[8]['c'] * args.width_multiplier), 1, self.network_settings[8]['s'],
                           args.dropout_prob, self.bias))

        ###############################################################################################################

        # Classification part
        self.network.append(nn.AvgPool2d((args.img_height // 32, args.img_width // 32)))
        self.network.append(nn.Dropout2d(args.dropout_prob, inplace=True))
        self.network.append(
            nn.Conv2d(int(self.network_settings[8]['c'] * args.width_multiplier), self.num_classes, 1, bias=self.bias))

        self.network = nn.Sequential(*self.network)

        self.initialize()

    def forward(self, x):
        x = self.network(x)
        x = x.view(-1, self.num_classes)
        return F.log_softmax(x)

    def initialize(self):
        """Initializes the model parameters"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias, self.bias_init)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)
