import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.input_tensor = None

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        return np.sum(-np.log(np.einsum('ij,ij->i', input_tensor, label_tensor) + np.finfo(np.float).eps))

    def backward(self, label_tensor):
        return -label_tensor / self.input_tensor

# test your implementation using cmdline param TestCrossEntropyLoss
