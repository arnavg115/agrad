import math
from .tensor import Tensor
import numpy as np
from typing import List


def relu(t: "Tensor"):
    out = Tensor(np.maximum(t.data, 0), (t,), "relu")

    def _backward():
        t.grad += np.multiply(out.grad, t.data > 0)

    out._backward = _backward
    return out


def exp(t: "Tensor"):
    out = Tensor(np.exp(t.data), (t,), "relu")

    def _backward():
        t.grad += np.exp(t.data) * out.grad

    out._backward = _backward
    return out


def tanh(t: "Tensor"):
    out = Tensor(np.tanh(t.data), (t,), "tanh")

    def _backward():
        t.grad += np.multiply(out.grad, 1 / (np.cosh(t.data) ** 2))

    out._backward = _backward
    return out


def sigmoid(t: "Tensor"):
    out = Tensor(1 / (1 + np.exp(-t.data)), (t,), "sigmoid")

    def _backward():
        t.grad += (t.data * (1 - t.data)) * out.grad

    out._backward = _backward
    return out


def leaky_relu(t: "Tensor", alpha=0.01):
    out = Tensor(np.maximum(t.data, t.data * alpha), (t,), "leakyrelu")

    def _backward():
        gr = np.ones_like(t.data)
        gr[t.data < 0] = alpha
        t.grad += np.multiply(gr, out.grad)

    out._backward = _backward
    return out


def log(t: "Tensor", base=math.e):
    out = Tensor(np.log(t.data) / np.log(base), (t,), "log")

    def _backward():
        gr = 1 / (t.data * np.log(base))
        t.grad += np.multiply(out.grad, gr)

    out._backward = _backward
    return out


def softmax(t: "Tensor", axis=-1, keepdims=True):
    a = exp(t)
    return a / a.sum(axis, keepdims)


def sqrt(t: "Tensor"):
    return t**0.5


def mean(t: "Tensor", axis=None, keepdims=True):
    if axis:
        return t.sum(axis, keepdims) / t.shape[axis]
    return t.sum() / t.size


def stack(l: List["Tensor"], axis=0):
    data = np.stack([ls.data for ls in l], axis)
    out = Tensor(data, tuple(l), "stack", l[0].req_grad)

    def _backward():
        sp = np.split(out.grad, len(l), axis)
        for ls, sps in zip(l, sp):
            if ls.req_grad:
                ls.grad += sps.reshape(ls.shape)

    out._backward = _backward
    return out


def silu(x):
    return x * sigmoid(x)


def sin(x: "Tensor"):
    out = Tensor(np.sin(x.data), (x), "sin", req_grad=x.req_grad)

    def _backward():
        x.grad += np.multiply(out.grad, np.cos(x.data))

    out._backward = _backward
    return out


def cos(x: "Tensor"):
    out = Tensor(np.sin(x.data), (x), "cos", req_grad=x.req_grad)

    def _backward():
        x.grad += np.multiply(out.grad, -np.sin(x.data))

    out._backward = _backward
    return out


def tan(x: "Tensor"):
    out = Tensor(np.tan(x.data), (x), "tan", req_grad=x.req_grad)

    def _backward():
        x.grad += np.multiply(out.grad, (1 / np.cos(x)) ** 2)

    out._backward = _backward
    return out


def gelu(x: "Tensor"):
    return 0.5 * x * (1 + tanh(math.sqrt(2 / math.pi) * (x + 0.044715 + x**3)))
