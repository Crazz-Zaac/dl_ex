import numpy as np
from typing import Union
from Layers.Base import BaseLayer

class Conv(BaseLayer):
    
    def __init__(
        self, stride_shape: Union[tuple, int], convolution_shape: tuple, num_kernels: int
    ) -> None:
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.weights = None
        self.bias = None
    
    
    def initialize(self, input_shape: tuple) -> None:
        self.input_shape = input_shape
        self.weights = np.random.randn(
            self.num_kernels, self.convolution_shape[0], self.convolution_shape[1]
        )
        self.bias = np.random.randn(self.num_kernels)
        self.output_shape = self.compute_output_shape()
