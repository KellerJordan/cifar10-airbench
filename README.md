# CIFAR-10 Airbench 💨

This repo contains utilities and baselines for fast neural network training on CIFAR-10.

| Script | Mean accuracy | Time | PFLOPs |
| - | - | - | - |
| `airbench94_compiled.py` | 94.01% | 3.29s | 0.36 |
| `airbench94.py` | 94.01% | 3.83s | 0.36 |
| `airbench95.py` | 95.01% | 10.4s | 1.4 |
| `airbench96.py` | 96.05% | 46.3s | 7.5 |

## How to run

```
git clone https://github.com/KellerJordan/cifar10-airbench.git && cd airbench && python airbench94.py
```

or

```
pip install airbench && python -c "import airbench; airbench.train94()"
```

