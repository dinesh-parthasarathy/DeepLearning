import numpy as np


class Sgd:  # Stochastic Gradient Descent
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - self.learning_rate * gradient_tensor


class SgdWithMomentum:
    def __init__(self, learning_rate: float, momentum):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v = self.momentum * self.v - self.learning_rate * gradient_tensor
        return weight_tensor + self.v


class Adam:
    def __init__(self, learning_rate: float, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = 0
        self.r = 0
        self.bias_v = 0
        self.bias_r = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1 - self.rho) * gradient_tensor ** 2
        self.bias_v = 1 - self.mu * (1 - self.bias_v)
        self.bias_r = 1 - self.rho * (1 - self.bias_r)

        return weight_tensor - self.learning_rate * (self.v / self.bias_v) / (np.sqrt(self.r / self.bias_r) + np.finfo(np.float).eps)

# verify your implementation using cmdline param TestOptimizers
