from ..module import Module
import numpy as np


class optim:
    def __init__(self, lr: float) -> None:
        self.lr = lr


class sgd(optim):
    def __init__(self, lr: float = 0.01) -> None:
        super().__init__(lr)

    def step(self, module: Module) -> None:
        for parameter in module.parameters():

            parameter.data -= parameter.grad * self.lr


class momentum(optim):
    def __init__(self, lr=0.01, beta=0.9) -> None:
        super().__init__(lr)
        self.beta = beta
        self.v_curr = {}

    def step(self, module: Module) -> None:
        for i in module.state_dict():
            if self.v_curr.get(i[0]) is None:
                self.v_curr[i[0]] = 0
            self.v_curr[i[0]] = (
                self.beta * self.v_curr[i[0]] + (1 - self.beta) * i[1].grad
            )
            i[1].data -= self.v_curr[i[0]] * self.lr


class rmsprop(optim):
    def __init__(self, lr: float = 0.01, beta=0.95, epsilon=1e-8) -> None:
        super().__init__(lr)
        self.beta = beta
        self.epsilon = epsilon
        self.s_curr = {}

    def step(self, module: Module) -> None:
        for i in module.state_dict():
            if self.s_curr.get(i[0]) is None:
                self.s_curr[i[0]] = 0
            self.s_curr[i[0]] = self.beta * self.s_curr[i[0]] + (
                1 - self.beta
            ) * np.power(i[1].grad, 2)
            d = i[1].grad / np.sqrt(self.s_curr[i[0]] + self.epsilon)
            i[1].data -= d * self.lr


class adam(optim):
    def __init__(self, lr: float = 0.01, beta1=0.9, beta2=0.9, epsilon=1e-8) -> None:
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.s_curr = {}
        self.v_curr = {}
        self.epsilon = epsilon

    def step(self, module: Module):
        for i in module.state_dict():
            if self.s_curr.get(i[0]) is None:
                self.s_curr[i[0]] = 0
            if self.v_curr.get(i[0]) is None:
                self.v_curr[i[0]] = 0
            self.v_curr[i[0]] = (
                self.beta2 * self.v_curr[i[0]] + (1 - self.beta2) * i[1].grad
            )
            self.s_curr[i[0]] = self.beta1 * self.s_curr[i[0]] + (
                1 - self.beta1
            ) * np.power(i[1].grad, 2)
            d = self.v_curr[i[0]] / np.sqrt(self.s_curr[i[0]] + self.epsilon)
            i[1].data -= d * self.lr
