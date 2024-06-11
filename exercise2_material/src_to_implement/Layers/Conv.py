import numpy as np
from typing import Union
from Layers.Base import BaseLayer


class Conv(BaseLayer):
    """
    Performs a convolution operation on the input tensor.
    
    Attributes:
        stride_shape: Union[tuple, int]
            The stride shape for the convolution operation.
        convolution_shape: tuple or int
            The shape of the convolution operation is [c, m, n] for 2D convolution and [c, m] for 1D convolution.
            c = number of input channels, m = height of the kernel, n = width of the kernel.
        num_kernels: int
            The number of kernels to be used in the convolution operation.
        weights: np.ndarray
            The weights for the convolution operation.
        bias: np.ndarray
            The bias for the convolution operation.
        _gradient_weights: np.ndarray
            The gradient of the weights for the convolution operation.
        _bias: np.ndarray
            The gradient of the bias for the convolution operation.
    
    Methods:
    
    """
    

    def __init__(
        self,
        stride_shape: Union[tuple, int],
        convolution_shape: tuple,
        num_kernels: int,
    ) -> None:
        super().__init__()
        self.trainable = True
        # Check if stride_shape is an integer or tuple
        self.stride_shape = stride_shape if isinstance(stride_shape, tuple) else (stride_shape, stride_shape)
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels # also called num_filters
        
        # Initialize weights for 1D and 2D convolution
        if len(convolution_shape) == 2:
            self.is1D = True
            self.weights = np.random.randn(
                num_kernels, convolution_shape[0], convolution_shape[1])
        if len(convolution_shape) == 3:
            self.is2D = True
            self.weights = np.random.randn(
                num_kernels, convolution_shape[0], convolution_shape[1], convolution_shape[2])
            
        self._gradient_weights = None
        self._bias = np.random.randn(num_kernels)

    @property
    def gradient_weights(self) -> np.ndarray:
        return self._gradient_weights

    @property
    def bias(self) -> np.ndarray:
        return self._bias
    
    
    # calculate the output shape of the convolutional layer
    def calculate_output_shape(self, input_shape: tuple) -> tuple:
        """
        Calculates the output shape of the convolutional layer.

        Args:
            input_shape: tuple
                The shape of the input tensor.

        Returns:
            tuple
                The shape of the output tensor.
        """
        # For 1D: input_shape = batch_size(b), channels(c), spatial_dim(y)
        if is1D:
            output_shape = 
        
        return output_shape
    
    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass for a convolutional layer.

        Args:
            input_tensor: np.ndarray
                The input tensor for the forward pass.

        Returns:
            np.ndarray
                The output tensor after applying the convolutional layer.
        """
        output_shape = 
        
