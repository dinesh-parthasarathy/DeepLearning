import copy


class NeuralNetwork:

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None

    def forward(self):
        output_prev, self.label_tensor = self.data_layer.next()

        for lyr in self.layers:
            output_prev = lyr.forward(output_prev)

        return self.loss_layer.forward(output_prev, self.label_tensor)

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for lyr in reversed(self.layers):
            error_tensor = lyr.backward(error_tensor)

    def append_trainable_layer(self, layer):
        opt = copy.deepcopy(self.optimizer)
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
