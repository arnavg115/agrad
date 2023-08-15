import inspect
from typing import Union
from .tensor import Tensor

class Module:
    def parameters(self):
        for n, v in inspect.getmembers(self):
            if isinstance(v,Tensor):
                yield v
            
            if isinstance(v, Module):
                yield from v.parameters()
    
    def zero_grad(self):
        for i in self.parameters():
            i.zero_grad()
    
    def __call__(self, *args) -> Tensor:
        return self.forward(*args)
    
    def _state_dict(self, pre = ""):
        for n,v in inspect.getmembers(self):
            if isinstance(v, Tensor):
                yield (pre+n,v)
            elif isinstance(v, Module):
                yield from v._state_dict(n+".")
        
    def state_dict(self):
        return list(self._state_dict())
            