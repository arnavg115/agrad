# Agrad - Auto Grad

A homecooked autogradient implementation using only Numpy. This is an extension of my ml library and I wish to use it to implement more complex networks. It is an amalgamation of Joel Grus' and Andrej Karpathy's implementation of autograd.

# Done:

- loss: mse, softmax cross-entropy
- optimizer: basic (SGD), adam, RMSprop, momentum
- activation/functions: tanh, leaky relu, sigmoid, relu, exp, basic tensor ops (add, subtract, matmul etc.)
- architectures: Linear (MLP) see mnist.py, Transformers (LLaMA.py)
- [RL](https://github.com/arnavg115/basic_rl): DQN, REINFORCE

# Todo:

- **GPU SUPPORT** (Using cupy or jax)
- building blocks: conv
- architectures: imagenet, Mamba
- other: KV cache for transformers, test backprop stability on transformer
