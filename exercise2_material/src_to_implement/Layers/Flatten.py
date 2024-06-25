import numpy as np
from Layers.Base import BaseLayer

class Flatten(BaseLayer):
    """
    It flattens the input tensor into a 2D tensor.
    
    Args:
        input_tensor: np.ndarray
            The input tensor to flatten.
            
    Returns:
        np.ndarray
            The flattened tensor.
    """

    def __init__(self) -> None:
        super().__init__()
        self.trainable = False

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        self.input_tensor_shape = input_tensor.shape
        # Flatten the input tensor into a 2D tensor with shape (batch_size, -1)
        return input_tensor.flatten().reshape((input_tensor.shape[0], -1))

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        # Reshape the error tensor back to the shape of the input tensor
        return error_tensor.reshape(self.input_tensor_shape)
