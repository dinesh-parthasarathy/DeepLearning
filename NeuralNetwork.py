class NeuralNetwork:
    # implement 5 public members
    # an optimizer object, loss: list, layers: list, data_layer, loss_layer
    # they will be set within the unit tests

    def __init(self):
        pass

    def forward(self):
        pass
    # return the output of the last layer (i.e. loss layer) of the network

    def backward(self):
        pass
    # start from the loss layer and propogate back through the network.

    def append_trainable_layer(self, layer):
        pass

    def train(self, iterations):
        pass
    # trains the network for iterations and stores the loss for each iteration

    def test(self, input_tensor):
        pass
    # propagates the input tensor through the network and returns the prediction of the last layer.

