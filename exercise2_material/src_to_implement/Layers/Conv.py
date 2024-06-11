import numpy as np
from typing import Union
from Layers.Base import BaseLayer
from scipy import signal

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
        self.stride_shape = (
            stride_shape
            if isinstance(stride_shape, tuple)
            else (stride_shape, stride_shape)
        )
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels  # also called num_filters
        self.is1D = False
        self.is2D = False

        # Initialize weights for 1D and 2D convolution
        if len(convolution_shape) == 2:
            self.is1D = True
            self.weights = np.random.randn(
                num_kernels, convolution_shape[0], convolution_shape[1]
            )
        if len(convolution_shape) == 3:
            self.is2D = True
            self.weights = np.random.randn(
                num_kernels,
                convolution_shape[0],
                convolution_shape[1],
                convolution_shape[2],
            )

        self._gradient_weights = None
        self._bias = np.random.randn(num_kernels)

    @property
    def gradient_weights(self) -> np.ndarray:
        return self._gradient_weights

    @property
    def bias(self) -> np.ndarray:
        return self._bias
    
    def initialize(self, weights_initializer, bias_initializer):
        """
        Initializes the weights and bias for the convolutional layer.

        Args:
            weights_initializer: str
                The initialization method for the weights.
            bias_initializer: str
                The initialization method for the bias.
        """
        self.weights = self.initialize_weights(
            weights_initializer, self.weights.shape
        )
        self.bias = self.initialize_bias(bias_initializer, self.bias.shape)

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
        if self.is1D:
            batch_size, channels, input_length = input_shape
            kernel_length = self.convolution_shape[1]
            padding = (kernel_length - 1) // 2
            stride = self.stride_shape[0]
            output_length = ((input_length + 2 * padding - kernel_length) // stride) + 1
            output_shape = (batch_size, self.num_kernels, output_length)

        if self.is2D:
            batch_size, channels, input_height, input_width = input_shape
            kernel_height, kernel_width = (
                self.convolution_shape[1],
                self.convolution_shape[2],
            )
            padding_h, padding_w = (kernel_height - 1) // 2, (kernel_width - 1) // 2
            stride_h, stride_w = self.stride_shape
            output_height = (
                (input_height + 2 * padding_h - kernel_height) // stride_h
            ) + 1
            output_width = (
                (input_width + 2 * padding_w - kernel_width) // stride_w
            ) + 1
            output_shape = (batch_size, self.num_kernels, output_height, output_width)

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
        self.input_tensor = input_tensor
        output_shape = self.calculate_output_shape(input_tensor.shape)
        output_tensor = np.zeros(output_shape)

        if self.is1D:
            padded_input = np.pad(
                self.input_tensor, ((0, 0), (0, 0), (1, 1)), mode="constant"
            )
            batch_size, num_kernels, output_length = output_shape
            for batch in range(batch_size):
                for kernel in range(num_kernels):
                    for i in range(output_length):
                        region = padded_input[batch, :, i : i + 3]
                        output_tensor[batch, kernel, i] = np.sum(
                            region * self.weights[kernel]
                        ) + self.bias[kernel]
                        
        if self.is2D:
            padded_input = np.pad(
                self.input_tensor, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant"
            )
            batch_size, num_kernels, output_height, output_width = output_shape
            for batch in range(batch_size):
                for kernel in range(num_kernels):
                    for i in range(output_height):
                        for j in range(output_width):
                            region = padded_input[batch, :, i : i + 3, j : j + 3]
                            output_tensor[batch, kernel, i, j] = np.sum(
                                region * self.weights[kernel]
                            ) + self.bias[kernel]
        
        return output_tensor                
                            
    
    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        self._gradient_weights = np.zeros_like(self.weights)
        self._bias = np.zeros_like(self.bias)
        gradient_input = np.zeros_like(self.input_tensor)
        
        