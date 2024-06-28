import numpy as np
from Layers.Base import BaseLayer

class Sigmoid(BaseLayer):
    """
    Provides the methods forward and backward for a Sigmoid activation function.
    """
    
    
    def __init__(self):
        super().__init__()
        self.activation = None
    
    def forward(self, input_tensor:np.ndarray) -> np.ndarray:
        """
        Computes the forward pass for a sigmoid activation function.
        
        Args:
            input_tensor: np.ndarray
                The input tensor for the forward pass.
        
        Returns:
            np.ndarray
                The output tensor after applying the sigmoid activation function.
        """
        self.activation = 1 / (1 + np.exp(- input_tensor))
        return self.activation
    
    def backward(self, error_tensor:np.ndarray) -> np.ndarray:
        """
        Computes the backward pass for a sigmoid activation function.
        
        Args:
            error_tensor: np.ndarray
                The error tensor for the backward pass.
        
        Returns:
            np.ndarray
                The error tensor after applying the backward pass.
        """
        return error_tensor * (self.activation * (1 - self.activation))