class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self,input_tensor,label_tensor):
        pass
    # computes the loss value according to the CrossEntropy Loss formula accumulated over the batch

    def backward(self,label_tensor):
        pass
    # returns the error tensor the previous layer. The backpropagation starts here hence no error_tensor is needed.
    # Instead, we need the label_tensor

# test your implementation using cmdline param TestCrossEntropyLoss

