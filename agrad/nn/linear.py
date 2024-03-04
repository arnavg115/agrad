from ..module import Module
from ..utils import zeros, random, randn

class Linear(Module):
    def __init__(self, inpt, otpt, bias = True, kaiming=False, mean = 0.0, std=0.1) -> None:
        if kaiming:
            self.weight = random((inpt, otpt), kaiming= kaiming)
        else:
            self.weight = randn((inpt, otpt), mean, std)
        self.b = bias
        if bias:
            self.bias = zeros((1,otpt))
        
    def forward(self, X):
        if self.b:
            return (X @ self.weight) + self.bias
        return X @ self.weight