from Layers.Base import BaseLayer
from Layers.Helpers import compute_bn_gradients
import numpy as np
import copy


class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self._optimizer = None
        self._optimizer_bias = None
        self._gradient_gamma = None
        self._gradient_beta = None
        self.channels = channels
        self.input_tensor = None
        self.input_tensor_tilde = None
        self.weights = None
        self.bias = None
        self.shape1 = None
        self.shape2 = None
        self.alpha = 0.8
        self.mu = None
        self.mu_b = None
        self.sigma = None
        self.sigma_b = None
        self.initialize()

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def optimizer_bias(self):
        return self._optimizer_bias

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value
        self._optimizer_bias = copy.deepcopy(self._optimizer)

    @property
    def gradient_weights(self):
        return self._gradient_gamma

    @property
    def gradient_bias(self):
        return self._gradient_beta

    def initialize(self, weights_init=None, bias_init=None):
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)

    def forward(self, input_tensor):
        reformat = False
        self.input_tensor = input_tensor
        if len(np.shape(input_tensor)) > 2:  # it is an output from the CNN layer, so reformat
            reformat = True
            input_tensor = self.reformat(input_tensor)

        if not self.testing_phase:
            self.mu_b = np.average(input_tensor, axis=0)
            self.sigma_b = np.std(input_tensor, axis=0)
            if self.mu is None:
                self.mu = self.mu_b
                self.sigma = self.sigma_b
            else:
                self.mu = self.alpha * self.mu + (1 - self.alpha) * self.mu_b
                self.sigma = self.alpha * self.sigma + (1 - self.alpha) * self.sigma_b

            input_tensor = (input_tensor - self.mu_b) / np.sqrt((self.sigma_b ** 2) + np.finfo(np.float).eps)  # normalize the input tensor

        else:
            input_tensor = (input_tensor - self.mu) / np.sqrt((self.sigma ** 2) + np.finfo(np.float).eps)  # normalize the input tensor

        self.input_tensor_tilde = input_tensor
        if reformat:
            self.input_tensor_tilde = self.reformat(self.input_tensor_tilde)

        input_tensor = input_tensor * self.weights + self.bias

        if reformat:  # reformat back to the original shape
            input_tensor = self.reformat(input_tensor)

        return input_tensor

    def backward(self, error_tensor):
        reformat = False
        if len(np.shape(self.input_tensor)) > 2:
            reformat = True
            self.input_tensor_tilde = self.reformat(self.input_tensor_tilde)
            self.input_tensor = self.reformat(self.input_tensor)
            error_tensor = self.reformat(error_tensor)
        self._gradient_gamma = np.sum(error_tensor * self.input_tensor_tilde, axis=0)
        self._gradient_beta = np.sum(error_tensor, axis=0)

        error_prev = compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.mu_b, self.sigma_b ** 2)

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self._gradient_gamma)
            self.bias = self.optimizer_bias.calculate_update(self.bias, self._gradient_beta)

        if reformat:
            self.input_tensor = self.reformat(self.input_tensor)
            self.input_tensor_tilde = self.reformat(self.input_tensor_tilde)
            error_prev = self.reformat(error_prev)

        return error_prev

    def reformat(self, tensor):
        if len(np.shape(tensor)) > 2:
            self.shape1 = np.shape(tensor)
            if len(self.shape1) is 4:
                tensor = np.reshape(tensor, (self.shape1[0], self.shape1[1], self.shape1[2] * self.shape1[3]))
            tensor = np.transpose(tensor, (0, 2, 1))
            self.shape2 = np.shape(tensor)
            tensor = np.reshape(tensor, (self.shape2[0] * self.shape2[1], self.shape2[2]))
        else:
            tensor = np.reshape(tensor, self.shape2)
            tensor = np.transpose(tensor, (0, 2, 1))
            tensor = np.reshape(tensor, self.shape1)
        return tensor
