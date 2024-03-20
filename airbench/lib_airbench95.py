# A variant of airbench optimized for time-to-95%.
# 10.8s runtime on an A100; 1.39 PFLOPs.
# Evidence: 95.01 average accuracy in n=200 runs.
# If random flip is used instead of alternating, then decays to 94.95 average accuracy in n=100 runs.
# With random flip and 16 epochs instead of 15, we get 94.97 in n=100 runs.
# With random flip and 17, we get 95.01 in n=100 runs.
#
# Changes relative to airbench:
# - Increased width and reduced learning rate.
# - Increased training duration to 15 epochs.

from .utils import train, evaluate, CifarLoader

#############################################
#            Setup/Hyperparameters          #
#############################################

import torch
from torch import nn
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True

# We express the main training hyperparameters (batch size, learning rate, momentum, and weight decay)
# in decoupled form, so that each one can be tuned independently. This accomplishes the following:
# * Assuming time-constant gradients, the average step size is decoupled from everything but the lr.
# * The size of the weight decay update is decoupled from everything but the wd.
# In constrast, normally when we increase the (Nesterov) momentum, this also scales up the step size
# proportionally to 1 + 1 / (1 - momentum), meaning we cannot change momentum without having to re-tune
# the learning rate. Similarly, normally when we increase the learning rate this also increases the size
# of the weight decay, requiring a proportional decrease in the wd to maintain the same decay strength.
#
# The practical impact is that hyperparameter tuning is faster, since this parametrization allows each
# one to be tuned independently. See https://myrtle.ai/learn/how-to-train-your-resnet-5-hyperparameters/.

hyp = {
    'opt': {
        'train_epochs': 15.0,
        'batch_size': 1024,
        'lr': 10.0,                 # learning rate per 1024 examples
        'momentum': 0.85,
        'weight_decay': 0.0153,     # weight decay per 1024 examples (decoupled from learning rate)
        'bias_scaler': 64.0,        # scales up learning rate (but not weight decay) for BatchNorm biases
        'label_smoothing': 0.2,
        'whiten_bias_epochs': 3,    # how many epochs to train the whitening layer bias before freezing
    },
    'aug': {
        'flip': True,
        'translate': 2,
    },
    'net': {
        'whitening': {
            'kernel_size': 2,
        },
        'batchnorm_momentum': 0.6,
        'base_width': 64,
        'scaling_factor': 1/9,
        'tta_level': 2,         # the level of test-time augmentation: 0=none, 1=mirror, 2=mirror+translate
    },
}

#############################################
#            Network Components             #
#############################################

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Mul(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, x):
        return x * self.scale

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-12, momentum=hyp['net']['batchnorm_momentum'],
                 weight=False, bias=True):
        super().__init__(num_features, eps=eps, momentum=1-momentum)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias
        # Note that PyTorch already initializes the weights to one and bias to zero

class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', bias=False):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        # Create an implicit residual via identity initialization
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in,  channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x

#############################################
#            Network Definition             #
#############################################

def make_net95():
    widths = {
        'block1': (2 * hyp['net']['base_width']), # 128 w/ width at base value
        'block2': (6 * hyp['net']['base_width']), # 384 w/ width at base value
        'block3': (6 * hyp['net']['base_width']), # 384 w/ width at base value
    }
    whiten_conv_width = 2 * 3 * hyp['net']['whitening']['kernel_size']**2
    net = nn.Sequential(
        Conv(3, whiten_conv_width, kernel_size=hyp['net']['whitening']['kernel_size'], padding=0, bias=True),
        nn.GELU(),
        ConvGroup(whiten_conv_width, widths['block1']),
        ConvGroup(widths['block1'],  widths['block2']),
        ConvGroup(widths['block2'],  widths['block3']),
        nn.MaxPool2d(3),
        Flatten(),
        nn.Linear(widths['block3'], 10, bias=False),
        Mul(hyp['net']['scaling_factor']),
    )
    net[0].weight.requires_grad = False
    net = net.half().cuda()
    net = net.to(memory_format=torch.channels_last)
    for mod in net.modules():
        if isinstance(mod, BatchNorm):
            mod.float()
    return net

############################################
#             Train and Eval               #
############################################

def train95(train_loader=CifarLoader('cifar10', train=True, batch_size=hyp['opt']['batch_size'], aug=hyp['aug'], altflip=True),
            epochs=hyp['opt']['train_epochs'], label_smoothing=hyp['opt']['label_smoothing'],
            learning_rate=hyp['opt']['lr'], bias_scaler=hyp['opt']['bias_scaler'],
            momentum=hyp['opt']['momentum'], weight_decay=hyp['opt']['weight_decay'],
            whiten_bias_epochs=hyp['opt']['whiten_bias_epochs'], tta_level=hyp['net']['tta_level'],
            make_net=make_net95, run=0, verbose=True):

    return train(train_loader, epochs, label_smoothing, learning_rate, bias_scaler, momentum, weight_decay,
                 whiten_bias_epochs, tta_level, make_net, run, verbose)

