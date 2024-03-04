from agrad import ones, F
import numpy as np
a = ones((2,2))
b = ones((2,2))
c = ones((2,2))
j = F.stack([a,b,c],0)

j.backward(np.arange(j.size).reshape(j.shape))
print(c.grad)