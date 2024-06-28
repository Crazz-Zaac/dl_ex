import numpy as np
from Layers.Base import BaseLayer

class Dropout(BaseLayer):
    """
    Dropout layer to be used in the neural network.
    
    Attributes:
        probability: float
            The dropout rate to apply
        mask: np.ndarray
            The mask to apply to the input tensor
    """
    
    def __init__(self, probability: float) -> None:
        super().__init__()
        
        self.probability = probability
        self.mask = None

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
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
        if self.testing_phase :
            return input_tensor 
        
        self.mask = np.random.rand(*input_tensor.shape) < self.probability
        
        return (input_tensor * self.mask) / self.probability
        
    
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
        return (error_tensor * self.mask) / (self.probability)
    
        
    
