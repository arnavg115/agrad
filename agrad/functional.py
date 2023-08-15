import math
from .tensor import Tensor
import numpy as np

def relu(t:"Tensor"):
    out = Tensor(np.maximum(t.data, 0), (t,), "relu")
    def _backward():
        t.grad += np.multiply(out.grad,t.data > 0)
    out._backward = _backward
    return out

def exp(t:"Tensor"):
    out = Tensor(np.exp(t.data), (t,), "relu")
    def _backward():
        t.grad += np.exp(t.data) * out.grad
    out._backward = _backward
    return out

def tanh(t:"Tensor"):
    out = Tensor(np.tanh(t.data), (t,),"tanh")
    
    def _backward():
        t.grad += np.multiply(out.grad, 1/(np.cosh(t.data) ** 2))
    out._backward = _backward
    return out

def sigmoid(t:"Tensor"):
    out = Tensor(1 / (1+ np.exp(-t.data)), (t,), "sigmoid")
    def _backward():
        t.grad += (t.data * (1-t.data)) * out.grad
    out._backward = _backward
    return out

def leaky_relu(t:"Tensor", alpha = 0.01):
    out = Tensor(np.maximum(t.data, t.data * alpha), (t,),"leakyrelu")
    def _backward():
        gr = np.ones_like(t.data)
        gr[t.data < 0] = alpha
        t.grad += np.multiply(gr, out.grad)
    
    out._backward = _backward
    return out

def log(t: "Tensor", base = math.e):
    out = Tensor(np.log(t.data)/np.log(base), (t,), "log")
    def _backward():
        gr =  1/(t.data * np.log(base))
        t.grad += np.multiply(out.grad, gr)
    out._backward = _backward
    return out