from ..module import Module
from ..utils import zeros, random

class Linear(Module):
    def __init__(self, inpt, otpt, bias = True, kaiming=True) -> None:
        self.weight = random((inpt, otpt), kaiming= kaiming)
        self.b = bias
        if bias:
            self.bias = zeros((1,otpt))
        
    def forward(self, X):
        if self.b:
            return (X @ self.weight) + self.bias
        return X @ self.weight