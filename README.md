# Agrad - Auto Grad/Arnav's Auto Grad

A substandard autogradient implementation using only Numpy. This is an extension of my ml library and I wish to use it to implement more complex networks. It is an amalgamation of Joel Grus' and Andrej Karpathy's implementation of autograd.

# Done:
- loss: mse
- optimizer: basic (SGD)
- activation/functions: relu, exp, basic tensor ops (add, subtract, matmul etc.)
- architectures: Linear (MLP) see mnist.py

# Todo:
- loss: SoftmaxCrossEntropy
- optimizer: momentum, adam, RMSprop
- metric: accuracy
- activation/functions: tanh, gelu, sin, cos, tan, logarithm
- architectures: LSTM, RNN
- other: Cleanup loss implementation and make it neater