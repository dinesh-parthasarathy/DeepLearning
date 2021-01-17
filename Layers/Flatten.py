import numpy as np
from Layers.Base import BaseLayer


class Flatten(BaseLayer):

    def __init__(self):
        super().__init__()
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.shape
        batch_size = int(np.shape(input_tensor)[0])
        dim = int(np.prod(input_tensor.shape) / batch_size)
        return input_tensor.reshape((batch_size, dim))

    def backward(self, error_tensor):
        return error_tensor.reshape(self.shape)
