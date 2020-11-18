import numpy as np


class FullyConnected:

    def __init__(self, input_size, output_size):
        self._optimizer = None
        self.input_tensor = None
        self._gradient_weights = None
        self.weights = np.random.rand(input_size + 1, output_size)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def gradient_weights(self):
        return self._gradient_weights

    def forward(self, input_tensor):
        input_tensor = np.append(input_tensor, np.ones((input_tensor.shape[0], 1)), axis=1)
        self.input_tensor = input_tensor
        return np.matmul(input_tensor, self.weights)

    def backward(self, error_tensor):
        weights_bp = self.weights[0:-1, 0:self.weights.shape[1]]  # remove bias entries for back propagation
        self._gradient_weights = np.matmul(np.transpose(self.input_tensor), error_tensor)

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self._gradient_weights)

        return np.matmul(error_tensor, np.transpose(weights_bp))

# verify your implementation using cmdline parameter TestFullyConnected
