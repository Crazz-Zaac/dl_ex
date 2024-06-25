import numpy as np
from copy import deepcopy

class Dropout:
    """
    Dropout layer to be used in the neural network.
    
    Attributes:
        probability: float
            The dropout rate to apply
        mask: np.ndarray
            The mask to apply to the input tensor
    """
    
    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.mask = None

    def forward(self, input_tensor: np.ndarray, phase: str) -> np.ndarray:
        """
        Forward pass for the dropout layer.
        
        Args:
            input_tensor: np.ndarray
                The input tensor for the forward pass
            phase: str
                The phase of the forward pass, 'train' or 'test'
                
        Returns:
            np.ndarray
                The output tensor
        """
        if phase == "train":
            self.mask = np.random.binomial(1, 1 - self.probability, size=input_tensor.shape)
            return input_tensor * self.mask
        else:
            return input_tensor # * (1 - self.probability)
    
    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        """
        Backward pass for the dropout layer.
        
        Args:
            error_tensor: np.ndarray
                The error tensor for the backward pass
                
        Returns:
            np.ndarray
                The error tensor for the next layer
        """
        return error_tensor * self.mask
    
        
    
