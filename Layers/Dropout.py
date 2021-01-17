from Layers.Base import BaseLayer
import numpy as np


class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.inv_prob = (1 / self.probability)
        self.rand = None

    def forward(self, input_tensor):
        if self.testing_phase:
            return input_tensor
        else:
            self.rand = np.random.random_sample(np.shape(input_tensor))
            return (self.rand < self.probability) * input_tensor * self.inv_prob

    def backward(self, error_tensor):
        if self.testing_phase:
            return error_tensor
        else:
            return (self.rand < self.probability) * error_tensor * self.inv_prob
