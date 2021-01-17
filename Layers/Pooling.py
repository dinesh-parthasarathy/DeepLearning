import numpy as np
from Layers.Base import BaseLayer


class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        # Similar to stride_shape and convolution_shape

    def forward(self, input_tensor):
        self.ip_shape = [input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]]
        # Hint: Keep in mind to store the correct information necessary for the backward pass
        # Different to the convolutional layer, the pooling layer must be implemented only for the 2D case
        dim1_shape = int(np.ceil((input_tensor.shape[2] - np.floor(self.pooling_shape[0] / 2)) / self.stride_shape[0]))
        dim2_shape = int(np.ceil((input_tensor.shape[3] - np.floor(self.pooling_shape[1] / 2)) / self.stride_shape[1]))

        self.ID = np.zeros_like(input_tensor)
        fwd = np.zeros((input_tensor.shape[0], input_tensor.shape[1], dim1_shape, dim2_shape), dtype='float')
        r = 0
        c = 0
        self.X_Max = np.zeros((input_tensor.shape[0], input_tensor.shape[1], dim1_shape, dim2_shape), dtype='int')
        self.Y_Max = np.zeros((input_tensor.shape[0], input_tensor.shape[1], dim1_shape, dim2_shape), dtype='int')

        for b in range(input_tensor.shape[0]):
            for ch in range(input_tensor.shape[1]):
                for row in range(0, input_tensor.shape[2], self.stride_shape[0]):
                    if (row + self.pooling_shape[0] <= input_tensor.shape[2]):
                        for col in range(0, input_tensor.shape[3], self.stride_shape[1]):
                            if (col + self.pooling_shape[1] <= input_tensor.shape[3]):
                                region = input_tensor[b, ch, row:self.pooling_shape[0] + row:1, col:self.pooling_shape[1] + col:1].copy()
                                result = np.where(region == np.amax(region))
                                id = list(zip(result[0], result[1]))
                                id_arr = np.array(id)
                                self.X_Max[b][ch][r][c] = id_arr[0][0]
                                self.Y_Max[b][ch][r][c] = id_arr[0][1]
                                fwd[b][ch][r][c] = input_tensor[b, ch, row + id_arr[0][0], col + id_arr[0][1]]
                                c = c + 1
                        c = 0
                        r = r + 1
                r = 0
                c = 0
        return fwd

    def backward(self, error_tensor):
        back = np.zeros(self.ip_shape, dtype='float')
        print('back.shape', back.shape)
        row_ID = 0
        col_ID = 0
        for b_e in range(error_tensor.shape[0]):
            for ch_e in range(error_tensor.shape[1]):
                for row_e in range(error_tensor.shape[2]):
                    for col_e in range(error_tensor.shape[3]):
                        val = error_tensor[b_e][ch_e][row_e][col_e]
                        x = int(self.X_Max[b_e][ch_e][row_e][col_e])
                        y = int(self.Y_Max[b_e][ch_e][row_e][col_e])
                        back[b_e, ch_e, row_ID + x, col_ID + y] += val
                        col_ID += self.stride_shape[1]
                    col_ID = 0
                    row_ID += self.stride_shape[0]
                row_ID = 0
                col_ID = 0
        return back
