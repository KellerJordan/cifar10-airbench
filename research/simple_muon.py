"""
Runs in ?? seconds on a 400W NVIDIA A100
Attains ?? mean accuracy (n=200 trials)
"""

#############################################
#            Setup/Hyperparameters          #
#############################################

import os
import sys
import uuid
from math import ceil

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from airbench import CifarLoader, evaluate

torch.backends.cudnn.benchmark = True

hyp = {
    'opt': {
        'train_epochs': 8,
        'batch_size': 2000,
        'lr': 6.5,                 # learning rate per 1024 examples
        'momentum': 0.85,
        'weight_decay': 0.015,     # weight decay per 1024 examples (decoupled from learning rate)
        'bias_scaler': 64.0,        # scales up learning rate (but not weight decay) for BatchNorm biases
        'label_smoothing': 0.2,
        'whiten_bias_epochs': 3,    # how many epochs to train the whitening layer bias before freezing
    },
    'aug': {
        'flip': True,
        'translate': 2,
    },
    'net': {
        'widths': {
            'block1': 64,
            'block2': 256,
            'block3': 256,
        },
        'batchnorm_momentum': 0.6,
        'scaling_factor': 1/9,
        'tta_level': 2,         # the level of test-time augmentation: 0=none, 1=mirror, 2=mirror+translate
    },
}

#############################################
#           Spectral SGD-momentum           #
#############################################

@torch.compile
def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if 'momentum_buffer' not in state.keys():
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum)

                p.data.mul_(len(p.data)**0.5 / p.data.norm()) # normalize the weight
                update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape) # whiten the update
                p.data.add_(update, alpha=-lr) # take a step

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
    def __init__(self, num_features, momentum, eps=1e-12,
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
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out, batchnorm_momentum):
        super().__init__()
        self.conv1 = Conv(channels_in,  channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out, batchnorm_momentum)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out, batchnorm_momentum)
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

def make_net():
    widths = hyp['net']['widths']
    batchnorm_momentum = hyp['net']['batchnorm_momentum']
    whiten_kernel_size = 2
    whiten_width = 2 * 3 * whiten_kernel_size**2
    net = nn.Sequential(
        Conv(3, whiten_width, whiten_kernel_size, padding=0, bias=True),
        nn.GELU(),
        ConvGroup(whiten_width,     widths['block1'], batchnorm_momentum),
        ConvGroup(widths['block1'], widths['block2'], batchnorm_momentum),
        ConvGroup(widths['block2'], widths['block3'], batchnorm_momentum),
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

def reinit_net(model):
    for m in model.modules():
        if type(m) in (Conv, BatchNorm, nn.Linear):
            m.reset_parameters()

#############################################
#       Whitening Conv Initialization       #
#############################################

def get_patches(x, patch_shape):
    c, (h, w) = x.shape[1], patch_shape
    return x.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).float()

def get_whitening_parameters(patches):
    n,c,h,w = patches.shape
    patches_flat = patches.view(n, -1)
    est_patch_covariance = (patches_flat.T @ patches_flat) / n
    eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO='U')
    return eigenvalues.flip(0).view(-1, 1, 1, 1), eigenvectors.T.reshape(c*h*w,c,h,w).flip(0)

def init_whitening_conv(layer, train_set, eps=5e-4):
    patches = get_patches(train_set, patch_shape=layer.weight.data.shape[2:])
    eigenvalues, eigenvectors = get_whitening_parameters(patches)
    eigenvectors_scaled = eigenvectors / torch.sqrt(eigenvalues + eps)
    layer.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))

############################################
#                 Logging                  #
############################################

def print_columns(columns_list, is_head=False, is_final_entry=False):
    print_string = ''
    for col in columns_list:
        print_string += '|  %s  ' % col
    print_string += '|'
    if is_head:
        print('-'*len(print_string))
    print(print_string)
    if is_head or is_final_entry:
        print('-'*len(print_string))

logging_columns_list = ['run   ', 'epoch', 'train_loss', 'train_acc', 'val_acc', 'tta_val_acc', 'total_time_seconds']
def print_training_details(variables, is_final_entry):
    formatted = []
    for col in logging_columns_list:
        var = variables.get(col.strip(), None)
        if type(var) in (int, str):
            res = str(var)
        elif type(var) is float:
            res = '{:0.4f}'.format(var)
        else:
            assert var is None
            res = ''
        formatted.append(res.rjust(len(col)))
    print_columns(formatted, is_final_entry=is_final_entry)

############################################
#                Training                  #
############################################

def main(run, model):

    batch_size = hyp['opt']['batch_size']
    epochs = hyp['opt']['train_epochs']
    momentum = hyp['opt']['momentum']
    # Assuming gradients are constant in time, for Nesterov momentum, the below ratio is how much
    # larger the default steps will be than the underlying per-example gradients. We divide the
    # learning rate by this ratio in order to ensure steps are the same scale as gradients, regardless
    # of the choice of momentum.
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    lr = hyp['opt']['lr'] / kilostep_scale # un-decoupled learning rate for PyTorch SGD
    wd = hyp['opt']['weight_decay'] * batch_size / kilostep_scale
    lr_biases = lr * hyp['opt']['bias_scaler']

    loss_fn = nn.CrossEntropyLoss(label_smoothing=hyp['opt']['label_smoothing'], reduction='none')

    test_loader = CifarLoader('cifar10', train=False, batch_size=2000)
    train_loader = CifarLoader('cifar10', train=True, batch_size=batch_size, aug=hyp['aug'])
    if run == 'warmup':
        # The only purpose of the first run is to warmup the compiled model, so we can use dummy data
        train_loader.labels = torch.randint(0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device)
    total_train_steps = ceil(len(train_loader) * epochs)

    # Reinitialize the network from scratch - nothing is reused from previous runs besides the PyTorch compilation
    reinit_net(model_trainbias)
    current_steps = 0

    # Create optimizers for train whiten bias stage
    filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if 'norm' in n and p.requires_grad]
    whiten_bias = model._orig_mod[0].bias
    fc_layer = model._orig_mod[-2].weight
    param_configs = [dict(params=norm_biases, lr=lr_biases, weight_decay=wd/lr_biases),
                     dict(params=[fc_layer], lr=lr, weight_decay=wd/lr)]
    optimizer1 = Muon(filter_params, lr=0.24, momentum=0.6)
    optimizer2 = torch.optim.SGD(param_configs, momentum=hyp['opt']['momentum'], nesterov=True)
    optimizer3 = torch.optim.SGD([whiten_bias], lr=lr, weight_decay=wd/lr, momentum=hyp['opt']['momentum'], nesterov=True)
    optimizers = [optimizer1, optimizer2, optimizer3]
    # Make learning rate schedulers for all 5 optimizers
    def get_lr(step):
        return 1 - step / total_train_steps
    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

    # For accurately timing GPU code
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    total_time_seconds = 0.0

    # Initialize the whitening layer using training images
    starter.record()
    train_images = train_loader.normalize(train_loader.images[:5000])
    init_whitening_conv(model_trainbias._orig_mod[0], train_images)
    ender.record()
    torch.cuda.synchronize()
    total_time_seconds += 1e-3 * starter.elapsed_time(ender)

    for epoch in range(ceil(epochs)):

        ####################
        #     Training     #
        ####################

        starter.record()

        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels).sum()
            model.zero_grad(set_to_none=True)
            loss.backward()
            for opt, sched in zip(optimizers, schedulers):
                opt.step()
                sched.step()
            current_steps += 1
            if current_steps >= total_train_steps:
                break

        ender.record()
        torch.cuda.synchronize()
        total_time_seconds += 1e-3 * starter.elapsed_time(ender)

        ####################
        #    Evaluation    #
        ####################

        # Save the accuracy and loss from the last training batch of the epoch
        train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
        train_loss = loss.item() / batch_size
        val_acc = evaluate(model, test_loader, tta_level=0)
        print_training_details(locals(), is_final_entry=False)
        run = None # Only print the run number once

    ####################
    #  TTA Evaluation  #
    ####################

    starter.record()
    tta_val_acc = evaluate(model, test_loader, tta_level=hyp['net']['tta_level'])
    ender.record()
    torch.cuda.synchronize()
    total_time_seconds += 1e-3 * starter.elapsed_time(ender)

    epoch = 'eval'
    print_training_details(locals(), is_final_entry=True)

    return tta_val_acc

if __name__ == "__main__":
    with open(sys.argv[0]) as f:
        code = f.read()

    model = make_net()
    model = torch.compile(model, mode='max-autotune')

    print_columns(logging_columns_list, is_head=True)
    main('warmup', model_trainbias, model_freezebias)
    accs = torch.tensor([main(run, model_trainbias, model_freezebias) for run in range(20)])
    print('Mean: %.4f    Std: %.4f' % (accs.mean(), accs.std()))

    log = {'code': code, 'accs': accs}
    log_dir = os.path.join('logs', str(uuid.uuid4()))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'log.pt')
    print(os.path.abspath(log_path))
    torch.save(log, os.path.join(log_dir, 'log.pt'))
