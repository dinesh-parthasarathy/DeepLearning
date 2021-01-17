import numpy as np
from Layers.Base import BaseLayer
import copy
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid
from Layers.FullyConnected import FullyConnected


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.ht = np.zeros(hidden_size)
        self.weights = np.random.rand(input_size + hidden_size + 1, hidden_size)
        self._memorize = False
        self._optimizer = None
        self._optimizer_FCN = None
        self._gradient_weights = None
        self.grad_FCNWeights = None
        self.ht_tensor = None
        self.output_tensor = None
        self.input_tensor = None
        self.tanH = TanH()
        self.sig = Sigmoid()
        self.FCN = FullyConnected(hidden_size, output_size)

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value
        self._optimizer_FCN = copy.deepcopy(self._optimizer)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    def calculate_regularization_loss(self):
        return self.optimizer.regularizer.norm(self.weights) + self._optimizer_FCN.regularizer.norm(self.FCN.weights)

    def initialize(self, weights_initializer, bias_initializer):
        input_size = np.shape(self.weights)[0] - 1
        output_size = np.shape(self.weights)[1]
        shape = (input_size, output_size)

        self.weights = np.append(weights_initializer.initialize(shape, input_size, output_size),
                                 bias_initializer.initialize((1, output_size), input_size, output_size), axis=0)
        self.FCN.initialize(weights_initializer, bias_initializer)

    def forward(self, input_tensor):

        time_steps = np.shape(input_tensor)[0]
        self.input_tensor = input_tensor
        self.output_tensor = np.zeros((time_steps, self.output_size))
        self.ht_tensor = np.zeros((time_steps, self.hidden_size))

        # Reset hidden state vector to 0
        if not self.memorize:
            self.ht = np.zeros(self.hidden_size)

        # Loop through time dimension
        for i in range(time_steps):
            # compute the new hidden state
            xt = input_tensor[i]
            xt = np.append(xt, self.ht)
            xt = np.append(xt, 1)
            self.ht = xt.dot(self.weights)
            self.ht = self.tanH.forward(self.ht)
            self.ht_tensor[i] = self.ht

            # compute the output vector
            yt = self.FCN.forward(self.ht)
            yt = self.sig.forward(yt)
            self.output_tensor[i] = yt

        return self.output_tensor

    def backward(self, error_tensor):

        time_steps = np.shape(error_tensor)[0]
        grad_ht = np.zeros((time_steps, self.hidden_size))
        error_i = np.zeros((time_steps, self.input_size))
        self.gradient_weights = np.zeros((self.input_size + self.hidden_size + 1, self.hidden_size))
        self.grad_FCNWeights = np.zeros((self.hidden_size + 1, self.output_size))

        for i in reversed(range(time_steps)):
            et = error_tensor[i]
            self.sig.activation = self.output_tensor[i]
            et = self.sig.backward(et)
            self.FCN.input_tensor = self.ht_tensor[i]
            self.grad_FCNWeights += (np.append(self.ht_tensor[i], 1).reshape(self.hidden_size + 1, 1) * et.reshape(1, self.output_size))
            et = self.FCN.backward(et) + grad_ht[i]
            self.tanH.activation = self.ht_tensor[i]
            et = self.tanH.backward(et)
            if i > 0:
                self.gradient_weights += (np.append(np.append(self.input_tensor[i], self.ht_tensor[i - 1]), 1).reshape(self.input_size + self.hidden_size + 1, 1)
                                          * et.reshape(1, np.shape(et)[0]))
            else:
                self.gradient_weights += (np.append(np.append(self.input_tensor[i], np.zeros(self.hidden_size)), 1).reshape(self.input_size + self.hidden_size + 1, 1)
                                          * et.reshape(1, np.shape(et)[0]))

            weights_bp = self.weights[0:-1, 0:self.weights.shape[1]]  # remove bias entries for back propagation

            et = et.dot(np.transpose(weights_bp))
            error_i[i] = et[:self.input_size]
            if i > 0:
                grad_ht[i - 1] = et[self.input_size:]

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.FCN.weights = self._optimizer_FCN.calculate_update(self.FCN.weights, self.grad_FCNWeights)

        return error_i
