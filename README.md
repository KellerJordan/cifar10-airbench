# CIFAR-10 Airbench 💨

Fast training baselines for CIFAR-10.


## How to run

To train a neural network with 94% accuracy, please run either

```
git clone https://github.com/KellerJordan/cifar10-airbench.git
python airbench/airbench94.py
```

or

```
pip install airbench
python -c "import airbench; airbench.train94()"
```


## Motivation

CIFAR-10 is among the most widely used datasets in machine learning, facilitating thousands of research projects per year. 
The goal of this repo is to provide researchers with fast and stable training baselines, in order to accelerate small-scale neural network research.
This repo contains three training baselines for CIFAR-10 which reach 94%, 95%, and 96% accuracy in state-of-the-art time.
These trainings are provided as easily runnable dependency-free PyTorch scripts, and can replace classic baselines like training ResNet-20 or ResNet-18.


## Training methods

| Script | Mean accuracy | Time | PFLOPs |
| - | - | - | - |
| `airbench94_compiled.py` | 94.01% | 3.29s | 0.36 |
| `airbench94.py` | 94.01% | 3.83s | 0.36 |
| `airbench95.py` | 95.01% | 10.4s | 1.4 |
| `airbench96.py` | 96.05% | 46.3s | 7.5 |

Timings are on a single NVIDIA A100 GPU.
Note that the first run of training is slower due to GPU warmup.


## Using the GPU-accelerated dataloader independently

For writing custom fast CIFAR-10 training scripts, you may find GPU-accelerated dataloading useful:
```
import airbench
train_loader = airbench.CifarLoader('/tmp/cifar10', train=True, aug=dict(flip=True, translate=4, cutout=16), batch_size=500)
test_loader = airbench.CifarLoader('/tmp/cifar10', train=False, batch_size=1000)

for epoch in range(200):
    for inputs, labels in train_loader:
        # outputs = model(inputs)
        # loss = F.cross_entropy(outputs, labels)
        ...
```

If you wish to modify the data used for training, it can be done like so:
```
import airbench
train_loader = airbench.CifarLoader('/tmp/cifar10', train=True, aug=dict(flip=True, translate=4, cutout=16), batch_size=500)
mask = (train_loader.labels < 6) # (this is just an example, the mask can be anything)
train_loader.images = train_loader.images[mask]
train_loader.labels = train_loader.labels[mask]
print(len(train_loader)) # The loader now contains 30,000 images and has batch size 500, so this prints 60.
```

## Example data-selection experiment

Airbench can be used as a platform for experiments in data selection and active learning.
The following is an example experiment which demonstrates the classic result that low-confidence examples provide more training signal than random examples.
It runs in <20 seconds on an A100.

```
import torch
from airbench import train94, infer, evaluate, CifarLoader

net = train94(label_smoothing=0) # train this network without label smoothing to get a better confidence signal

loader = CifarLoader('cifar10', train=True, batch_size=1000)
logits = infer(net, loader)
conf = logits.log_softmax(1).amax(1) # confidence

train_loader = CifarLoader('cifar10', train=True, batch_size=1024, aug=dict(flip=True, translate=2))
mask = (torch.rand(len(train_loader.labels)) < 0.6)
print('Training on %d images selected randomly' % mask.sum())
train_loader.images = train_loader.images[mask]
train_loader.labels = train_loader.labels[mask]
train94(train_loader, epochs=16) # yields around 93% accuracy

train_loader = CifarLoader('cifar10', train=True, batch_size=1024, aug=dict(flip=True, translate=2))
mask = (conf < conf.float().quantile(0.6))
print('Training on %d images selected based on minimum confidence' % mask.sum())
train_loader.images = train_loader.images[mask]
train_loader.labels = train_loader.labels[mask]
train94(train_loader, epochs=16) # yields around 94% accuracy => low-confidence sampling is better than random.
```

