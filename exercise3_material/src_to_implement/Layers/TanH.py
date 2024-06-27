import numpy as np
from Layers.Layer import *

class TanH:
    """
    Provides the methods forward and backward for a TanH activation function.
    """
    
    
    def __init__(self):
        super().__init__()
        self.activation = None
    
    def forward(self, input_tensor:np.ndarray) -> np.ndarray:
        """
        Computes the forward pass for a hyperbolic tangent (TanH) activation function.
        
        Args:
            input_tensor: np.ndarray
                The input tensor for the forward pass.
        
        Returns:
            np.ndarray
                The output tensor after applying the TanH activation function.
        """
        self.activation = np.tanh(input_tensor)
        return self.activation
    
    def backward(self, error_tensor:np.ndarray) -> np.ndarray:
        """
        Computes the backward pass for a hyperbolic tangent (TanH) activation function.
        
        Args:
            error_tensor: np.ndarray
                The error tensor for the backward pass.
        
        Returns:
            np.ndarray
                The error tensor after applying the backward pass.
        """
        return error_tensor * (1 - self.activation ** 2)

