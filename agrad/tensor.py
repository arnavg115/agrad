import numpy as np

def broadcast(t: tuple, data:np.ndarray):
    nd = data.ndim - len(t)
    for i in range(nd):
        data = data.sum(axis=0)
    
    for j, dim in enumerate(t):
        if dim == 1:
            data = data.sum(axis=j,keepdims=True)
    return data


class Tensor:
    def __init__(self, data:np.ndarray, _children=(), op: str="", req_grad = True) -> None:
        self.data = data
        self.shape = data.shape
        if req_grad:
            self.grad = np.zeros_like(data)
        self._backward = None
        self._prev = set(_children)
        self._op = op
        self.req_grad = req_grad
        ()
    
    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def __neg__(self):
        out = Tensor(-1 * self.data, (self,), "neg", req_grad=self.req_grad)
        def _backward():
            if self.req_grad:
                self.grad += -1 * out.grad
        
        out._backward = _backward
        return out

    def __mul__(self, other:"Tensor"):
        out = Tensor(np.multiply(self.data, other.data), (self, other), "mul")
        
        def _backward():
            self.grad += broadcast(self.data.shape, np.multiply(out.grad, other.data))
            other.grad += broadcast(other.data.shape, np.multiply(out.grad, self.data))
        out._backward = _backward
        return out
    
    def sum(self, axis=None,keepdims=False):
        
        out = Tensor(np.array(self.data.sum(),),(self,), "sum")
        if axis is not None:
            out = Tensor(np.array(self.data.sum(axis=axis,keepdims=keepdims)),(self,), "sum")
        def _backward():
            self.grad += out.grad * np.ones_like(self.data)
        
        out._backward = _backward
        return out
    
    def __add__(self, other:"Tensor"):
        # self: (3,1) other: (3,3)
        # out: (3,3)
        
        out = Tensor(self.data + other.data, (self, other), "add", req_grad= self.req_grad or other.req_grad)
        
        def _backward():
            if self.req_grad:
                self.grad += broadcast(self.data.shape,out.grad)
            if other.req_grad:
                other.grad += broadcast(other.data.shape, out.grad)
        
        out._backward = _backward
        return out
    
    def __matmul__(self,other:"Tensor"):
        out = Tensor(self.data.dot(other.data), (self, other), "matmul", req_grad= self.req_grad or other.req_grad)
        def _backward():
            # self: (3,5) other: (5,4)
            # out: (3,4)
            if self.req_grad:
                self.grad += out.grad.dot(other.data.T)
            if other.req_grad:
                other.grad += self.data.T.dot(out.grad)
        out._backward = _backward
        return out

    def backward(self, grad: np.ndarray):
        if not self.req_grad:
            raise NotImplementedError
        netw:list[Tensor] = []
        alr = set()
        
        def build(t: Tensor):
            if t not in netw:
                alr.add(t)
                for child in t._prev:
                    build(child)
                netw.append(t)
        
        build(self)
        self.grad += grad
        for l in reversed(netw):
            if l._backward is not None:
                l._backward()
        
    
    def __sub__(self, other):
        return self + - other

    def __repr__(self) -> str:
        return f"Tensor({self.data})"
    
    def __pow__(self, exp):
        out = Tensor(np.power(self.data, exp), (self,), "pow", req_grad= self.req_grad)
        
        def _backward():
            if self.req_grad:
                self.grad += np.multiply(self.data ** (exp-1), exp) * out.grad
        
        out._backward = _backward
        return out

    def __getitem__(self,items):
        return Tensor(self.data[items])
    
    def __truediv__(self,other):
        return self * (other ** -1)
            
            # x / a
            # a / x: - a /x^2