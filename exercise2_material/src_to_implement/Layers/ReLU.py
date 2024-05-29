import numpy as np
from Layers.Base import BaseLayer


class ReLU(BaseLayer):
    """
    Provides the methods forward and backward for a ReLU activation function.
    """
    
    
    def __init__(self):
        super().__init__()
    
    def forward(self, input_tensor:np.ndarray) -> np.ndarray:
        """
        Computes the forward pass for a rectified linear unit (ReLU) activation function.
        
        Args:
            input_tensor: np.ndarray
                The input tensor for the forward pass.
        
        Returns:
            np.ndarray
                The output tensor after applying the ReLU activation function.
        """
        input_tensor[input_tensor < 0] = 0 
        self.output_tensor = input_tensor
        return input_tensor
    
    def backward(self, error_tensor:np.ndarray) -> np.ndarray:
        """
        Computes the backward pass for a rectified linear unit (ReLU) activation function.
        
        Args:
            error_tensor: np.ndarray
                The error tensor for the backward pass.
        
        Returns:
            np.ndarray
                The error tensor after applying the backward pass.
        """
        error_tensor[self.output_tensor <= 0] = 0
        return error_tensor
            