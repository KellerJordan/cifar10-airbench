from .lib_airbench93 import train93, make_net93
from .lib_airbench94 import train94, make_net94
from .lib_airbench95 import train95, make_net95
from .lib_airbench96 import train96, make_net96
from .utils import infer, evaluate, CifarLoader

def warmup93(*args, **kwargs):
    return train93(*args, run=-1, **kwargs)
def warmup94(*args, **kwargs):
    return train94(*args, run=-1, **kwargs)
def warmup95(*args, **kwargs):
    return train95(*args, run=-1, **kwargs)
def warmup96(*args, **kwargs):
    return train96(*args, run=-1, **kwargs)

