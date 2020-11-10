class FullyConnected:

    # add protected member _optimizer with setter and getter
    # add members weights and gradient_weights

    def __init__(self, input_size, output_size):
        pass
    # Initialize weights of this layer uniformly random in the range [0,1)

    def forward(self, input_tensor):
        pass
    # return a tensor which serves as the input tensor for the next layer

    def backward(self, error_tensor):
        pass
    # return a tensor which serves as the error tensor for the previous layer
    # use the method calculate_update of your optimizer, in order to update weights.
    # don't perform an update if the optimizer is not set

# verify your implementation using cmdline parameter TestFullyConnected
