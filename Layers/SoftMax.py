import numpy as np
from Layers.Base import BaseLayer


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.output_tensor = None

    def forward(self, input_tensor):
        input_tensor = (input_tensor.T - np.amax(input_tensor, axis=1)).T
        input_tensor = np.exp(input_tensor)
        self.output_tensor = (input_tensor.T / input_tensor.sum(axis=1)).T
        return self.output_tensor

    def backward(self, error_tensor):
        return np.multiply(self.output_tensor,
                           (error_tensor.T - np.einsum('ij,ij->i', error_tensor, self.output_tensor)).T)

# test your implementation using cmdline param TestSoftMax
