import copy


class NeuralNetwork:

    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.weight_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None
        self._phase = None

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        self._phase = value

    def forward(self):
        input_data, self.label_tensor = self.data_layer.next()
        regularization_loss = 0
        for lyr in self.layers:
            lyr.testing_phase = self.phase
            if lyr.weights is not None:
                if lyr.optimizer is not None:
                    if lyr.optimizer.regularizer is not None:
                        regularization_loss += lyr.optimizer.regularizer.norm(lyr.weights)

            input_data = lyr.forward(input_data)

        return self.loss_layer.forward(input_data, self.label_tensor) + regularization_loss

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for lyr in reversed(self.layers):
            error_tensor = lyr.backward(error_tensor)

    def append_trainable_layer(self, layer):
        opt = copy.deepcopy(self.optimizer)
        layer.initialize(self.weight_initializer, self.bias_initializer)
        layer.optimizer = opt
        self.layers.append(layer)

    def train(self, iterations):
        self.phase = False
        for i in range(iterations):
            self.loss.append(self.forward())
            self.backward()

    def test(self, input_tensor):
        self.phase = True
        for lyr in self.layers:
            lyr.testing_phase = self.phase
            input_tensor = lyr.forward(input_tensor)

        return input_tensor

    # propagates the input tensor through the network and returns the prediction of the last layer.
