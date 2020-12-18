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

    def forward(self):
        input_data, self.label_tensor = self.data_layer.next()

        for lyr in self.layers:
            input_data = lyr.forward(input_data)

        return self.loss_layer.forward(input_data, self.label_tensor)

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
        for i in range(iterations):
            self.loss.append(self.forward())
            self.backward()

    def test(self, input_tensor):
        for lyr in self.layers:
            input_tensor = lyr.forward(input_tensor)

        return input_tensor

    # propagates the input tensor through the network and returns the prediction of the last layer.
