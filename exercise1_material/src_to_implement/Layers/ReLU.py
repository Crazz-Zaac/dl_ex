import numpy as np


class ReLU:
    """
    Provides the methods forward and backward for a ReLU activation function.
    """
    
    
    def __init__(self):
        self.output_tensor = None
    
    def forward(self, input_tensor):
        """
        Computes the forward pass for a rectified linear unit (ReLU) activation function.
        
        Args:
            input_tensor: np.ndarray
                The input tensor for the forward pass.
        
        Returns:
            np.ndarray
                The output tensor after applying the ReLU activation function.
        """
        self.input_tensor = input_tensor
        return np.maximum(input_tensor, 0)
    
    def backward(self, error_tensor):
        """
        Computes the backward pass for a rectified linear unit (ReLU) activation function.
        
        Args:
            error_tensor: np.ndarray
                The error tensor for the backward pass.
        
        Returns:
            np.ndarray
                The error tensor after applying the backward pass.
        """
        return error_tensor * (self.input_tensor > 0)
            