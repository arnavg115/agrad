from agrad import ones, F
import numpy as np


a = ones((2,2,4))
w = ones((1,4))

b = a + w
print(b)
b.backward(np.ones_like(b,dtype=float))

print(w.grad)