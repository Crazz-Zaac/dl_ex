import numpy as np
from Layers.Base import *


class SoftMax(BaseLayer):
    """
    Transforms the input tensor into a tensor of probabilities using the softmax function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass for a softmax function.

        Args:
            input_tensor: np.ndarray
                The input tensor for the forward pass.

        Returns:
            np.ndarray
                The output tensor after applying the softmax function.
        """
        # shifting the input tensor to avoid numerical instability
        self.input_tensor = (
            input_tensor - input_tensor.max(axis=1, keepdims=True)
        )
        # 
        exp_input_tensor = np.exp(self.input_tensor)
        
        self.output_tensor = exp_input_tensor / exp_input_tensor.sum(
            axis=1, keepdims=True
        )
        return self.output_tensor

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        """
        Computes the backward pass for a softmax function.

        Args:
            error_tensor: np.ndarray
                The error tensor for the backward pass.

        Returns:
            np.ndarray
                The error tensor after applying the backward pass.
        """
        error = error_tensor * self.output_tensor
        error = error.sum(axis=1, keepdims=True)
        error = error_tensor - error
        error = self.output_tensor * error

        return error
