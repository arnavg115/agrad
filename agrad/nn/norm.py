from ..module import Module
from ..utils import ones
from ..tensor import Tensor
from .. import functional as F

class rmsnorm(Module):
  def __init__(self,axis,eps=1e-6):
    self.eps = eps
    self.weight = ones(axis)

  def forward(self, x:"Tensor"):
    x = x / F.sqrt(F.mean((x ** 2), axis=-1) + self.eps)
    return (x * self.weight)