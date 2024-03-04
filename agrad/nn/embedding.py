from agrad import Tensor
from ..module import Module
from ..utils import zeros, random, randn

class Embedding(Module):
    def __init__(self, inpt, otpt, bias = True, kaiming=True, mean=0.0, std=0.1) -> None:
        if kaiming:
            self.weight = random((inpt, otpt), kaiming=kaiming)
        else:
            self.weight = randn((inpt, otpt), mean, std)
        
        
    def forward(self, X):
        if isinstance(X,Tensor):
            return self.weight[X.data]
        return self.weight[X]