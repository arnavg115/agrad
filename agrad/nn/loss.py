from typing import Any
from ..tensor import Tensor
from ..utils import mean, exp, log
import numpy as np


class MSE:
    def __call__(self, predicted: Tensor, target: Tensor, reduction = "sum") -> Tensor:
        self.predicted = predicted
        self.target = target
        err = predicted - target
        if reduction == "sum":
            return (err * err).sum()
        return mean((err * err))


class _SoftmaxCrossEnt:
    def softmax(self, x:"Tensor"):
        exp_x = exp(x)
        return mean(exp_x, axis=1, keepdims=True)
    
    def __call__(self, predicted:Tensor, target:Tensor, reduction = "mean"):
        y_pred = self.softmax(predicted)
        loss = -1 * (target * log(y_pred)).sum(axis=0)
        return mean(loss)
        

class SoftmaxCrossEnt:
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def __call__(self, predicted: Tensor, target: Tensor) -> Any:
        self.predicted = predicted
        self.target = target
        y_pred = self.softmax(self.predicted.data)
        loss = -np.sum(target.data * np.log(y_pred), axis=0)
        return np.mean(loss)

    def backward(self):
        y_pred = self.softmax(self.predicted.data)
        gr = y_pred - self.target.data
        self.predicted.backward(gr)
