import numpy as np
from typing import Union
from Layers.Base import BaseLayer


class Conv(BaseLayer):

    def __init__(
        self,
        stride_shape: Union[tuple, int],
        convolution_shape: tuple,
        num_kernels: int,
    ) -> None:
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels # also called num_filters
        
        # Initialize weights for 1D and 2D convolution
        if len(convolution_shape) == 2:
            self.weights = np.random.randn(
                num_kernels, convolution_shape[0], convolution_shape[1])
        if len(convolution_shape) == 3:
            self.weights = np.random.randn(
                num_kernels, convolution_shape[0], convolution_shape[1], convolution_shape[2])
            
        self._gradient_weights = None
        self._bias = None

    @property
    def gradient_weights(self) -> np.ndarray:
        return self._gradient_weights

    @property
    def bias(self) -> np.ndarray:
        return self._bias
