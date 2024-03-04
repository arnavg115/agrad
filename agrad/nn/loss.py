from typing import Any
from ..tensor import Tensor
import numpy as np

class MSE:
    def __call__(self, predicted: Tensor, target: Tensor) -> Tensor:
        self.predicted = predicted
        # print(predicted.shape)
        # print(target.shape)
        self.target = target
        err = (predicted - target)
        return (err * err).sum().data
    
    def backward(self):
        gr = 2 * (self.predicted.data - self.target.data)
        self.predicted.backward(gr)


class SoftmaxCrossEnt:
    def softmax(self, x):
        exp_x = np.exp(x-np.max(x,axis=1,keepdims=True))
        return exp_x / np.sum(exp_x, axis=1,keepdims=True)
    
    def __call__(self,predicted: Tensor, target: Tensor) -> Any:
        self.predicted = predicted
        self.target = target
        y_pred = self.softmax(self.predicted.data)
        loss = -np.sum(target.data * np.log(y_pred), axis=0)
        return np.mean(loss)
    
    def backward(self):
        y_pred = self.softmax(self.predicted.data)
        gr = y_pred - self.target.data
        self.predicted.backward(gr)
