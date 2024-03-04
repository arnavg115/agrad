import math
import numpy as np
import pickle

from agrad.module import Module
from .tensor import Tensor

def one_hot_func(array, size):
    n = len(array)
    one_hot = np.zeros((n, size))
    one_hot[np.arange(n), array] =1
    return one_hot

def randn(shape: tuple, mean, std):
    return Tensor(np.random.normal(mean,std,shape))


def random(shape: tuple, kaiming = False):
    if kaiming:
        dt = (2 * np.random.rand(*shape) -1) * math.sqrt(shape[0])
        return Tensor(dt)
    return Tensor(np.random.randn(*shape))

def ones(shape: tuple, dtype=None):
    return Tensor(np.ones(shape,dtype=dtype))

def zeros(shape: tuple):
    return Tensor(np.zeros(shape))

accuracy = lambda x,y: (np.argmax(x, axis=1) == np.argmax(y, axis=1)).sum() / y.shape[0]

def save_model(file_name:str,module: Module):
    with open(file_name,"wb") as f:
        pickle.dump(module, f)

def load_model(file_name:str):
    with open(file_name, "rb") as f:
        return pickle.load(f)

def save_state_dict(file_name:str, module: Module):
    with open(file_name, "wb") as f:
        pickle.dump(module.state_dict(), f)

def load_state_dict(file_name: str, module:Module):
    f = open(file_name, "rb")
    sd:dict[str, Tensor] = dict(pickle.load(f))
    for n, t in module._state_dict():
        t.data = sd[n].data
    return module