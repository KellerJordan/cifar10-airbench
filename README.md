# CIFAR-10 Airbench 💨

This repo contains utilities and baselines for fast neural network training on CIFAR-10.


## How to run

To perfom a fast training targeting 94% accuracy, run either

```
git clone https://github.com/KellerJordan/cifar10-airbench.git && cd airbench && python airbench94.py
```

or

```
pip install airbench && python -c "import airbench; airbench.train94()"
```


## Using the GPU-accelerated dataloader independently

For your own fast CIFAR-10 training scripts, you may find GPU-accelerated dataloading useful:
```
import airbench
train_loader = airbench.CifarLoader('cifar10', train=True, aug=dict(flip=True, translate=4, cutout=16), batch_size=500)
test_loader = airbench.CifarLoader('cifar10', train=False, batch_size=1000)

for epoch in range(200):
    for inputs, labels in train_loader:
        # outputs = model(inputs)
        # loss = F.cross_entropy(outputs, labels)
        ...
```

If you wish to modify the data used for training, it can be done like so:
```
import airbench
train_loader = airbench.CifarLoader('cifar10', train=True, aug=dict(flip=True, translate=4, cutout=16), batch_size=500)
mask = (train_loader.labels < 6)
train_loader.images = train_loader.images[mask]
train_loader.labels = train_loader.labels[mask]
print(len(train_loader)) # The loader now contains 30,000 images and has batch size 500, so this prints 60.
```


## Motivation

CIFAR-10 is among the most widely used datasets in machine learning, facilitating thousands of research projects per year. 
However, many studies use poorly optimized trainings, leading to wasted time and sometimes contradictory results.
To resolve this problem, airbench contains a set of training methods which are both (a) very easily runnable and (b) state-of-the-art in terms of training speed.

In particular, airbench contains training scripts which achieve 94%, 95%, and 96% accuracy on the CIFAR-10 test-set in state-of-the-art time.
These methods can replace baselines like training ResNet-20 or ResNet-18.


## Training baselines

| Script | Mean accuracy | Time | PFLOPs |
| - | - | - | - |
| `airbench94_compiled.py` | 94.01% | 3.29s | 0.36 |
| `airbench94.py` | 94.01% | 3.83s | 0.36 |
| `airbench95.py` | 95.01% | 10.4s | 1.4 |
| `airbench96.py` | 96.05% | 46.3s | 7.5 |

Timings are on a single NVIDIA A100.
Note that the first run of training is always slower due to GPU warmup.

