import numpy as np
import copy
from scipy import signal
from Layers.Base import BaseLayer


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels: int):
        super().__init__()
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.weights = np.random.random_sample((num_kernels,) + convolution_shape)
        self.bias = np.random.random_sample((num_kernels,))
        self._optimizer = None
        self._optimizer_bias = None
        self._gradient_weights = np.zeros(np.shape(self.weights))
        self._gradient_bias = np.zeros(np.shape(self.bias))
        self.input_shape = None
        self.input_tensor = None
        if len(stride_shape) == 1 and len(convolution_shape) == 3:  # if stride shape is a single value for 2D convolution
            self.stride_shape = np.append(stride_shape, stride_shape, axis=0)

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
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.prod(self.convolution_shape)
        fan_out = self.num_kernels * np.prod(self.convolution_shape[1:])
        self.weights = weights_initializer.initialize(np.shape(self.weights), fan_in, fan_out)
        self.bias = bias_initializer.initialize(np.shape(self.bias), fan_in, fan_out)

    def forward(self, input_tensor):
        self.input_shape = np.shape(input_tensor)
        self.input_tensor = input_tensor
        output_tensor = np.zeros((self.input_shape[0],) + (self.num_kernels,) + tuple(int(np.ceil(dimension / stride)) for dimension, stride in zip(self.input_shape[2:], self.stride_shape)))
        batch_size = self.input_shape[0]
        no_channels = self.input_shape[1]
        for i in range(batch_size):
            for j in range(self.num_kernels):
                if len(self.convolution_shape) == 3:
                    output_tensor[i, j] = signal.correlate(input_tensor[i], self.weights[j], mode='same')[no_channels // 2][::self.stride_shape[0], ::self.stride_shape[1]] + self.bias[
                        j]  # 2D convolution
                else:
                    output_tensor[i, j] = signal.correlate(input_tensor[i], self.weights[j], mode='same')[no_channels // 2][::self.stride_shape[0]] + self.bias[j]  # 1D convolution

        return output_tensor

    def backward(self, error_tensor):
        error_tensor_prev = np.zeros(self.input_shape)
        no_input_channels = self.convolution_shape[0]
        batch_size = self.input_shape[0]

        # upsample error tensor
        error_tensor_upsampled = np.zeros(np.shape(error_tensor)[0:2] + self.input_shape[2:])
        if len(self.convolution_shape) == 3:
            error_tensor_upsampled[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor
        else:
            error_tensor_upsampled[:, :, ::self.stride_shape[0]] = error_tensor

        # gradient w.r.t bias
        self._gradient_bias = np.zeros(np.shape(self.bias))
        for b in range(batch_size):
            if len(self.convolution_shape) == 2:  # 1D convolution
                self._gradient_bias += np.sum(error_tensor[b], axis=1)
            else:  # 2D convolution
                self._gradient_bias += np.sum(np.sum(error_tensor[b], axis=1), axis=1)

        # gradient w.r.t weights
        input_tensor_padded = np.zeros(self.input_shape[0:2] + tuple(image_dim + kernel_dim - 1 for image_dim, kernel_dim in zip(self.input_shape[2:], self.convolution_shape[1:])))
        self._gradient_weights = np.zeros(np.shape(self.weights))
        pad_y_start = (self.convolution_shape[1:][0]) // 2
        pad_y_end = np.shape(input_tensor_padded)[2] - (self.convolution_shape[1:][0] - 1) // 2
        if len(self.convolution_shape) == 2:  # 1D convolution
            input_tensor_padded[:, :, pad_y_start:pad_y_end] = self.input_tensor
        else:
            pad_x_start = (self.convolution_shape[1:][1]) // 2
            pad_x_end = np.shape(input_tensor_padded)[3] - (self.convolution_shape[1:][1] - 1) // 2
            input_tensor_padded[:, :, pad_y_start:pad_y_end, pad_x_start:pad_x_end] = self.input_tensor

        for b in range(batch_size):
            for i in range(np.shape(error_tensor)[1]):
                for j in range(no_input_channels):
                    self._gradient_weights[i, j] += signal.correlate(input_tensor_padded[b, j], error_tensor_upsampled[b, i], mode='valid')

        # gradient w.r.t input
        weights_back = np.zeros((no_input_channels,) + (self.num_kernels,) + self.convolution_shape[1:])
        for i in range(no_input_channels):  # re arrange weights
            weights_stack = self.weights[0, i].reshape((1,) + self.convolution_shape[1:])
            for _ in (j + 1 for j in range(self.num_kernels - 1)):
                weights_stack = np.concatenate((weights_stack, self.weights[_, i].reshape((1,) + self.convolution_shape[1:])), axis=0)
            weights_back[i] = weights_stack
            if len(self.convolution_shape) == 3:
                weights_back[i] = np.flip(weights_stack, axis=0)
        for b in range(batch_size):
            for i in range(no_input_channels):
                error_tensor_prev[b, i] = signal.convolve(error_tensor_upsampled[b], weights_back[i], mode='same')[self.num_kernels // 2]

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self._gradient_weights)
            self.bias = self.optimizer_bias.calculate_update(self.bias, self._gradient_bias)

        return error_tensor_prev
