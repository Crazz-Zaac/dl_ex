import numpy as np


class Flatten:
    """
    Flattens the input tensor into a 1D tensor.
    """

    def __init__(self) -> None:
        pass

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        self.input_tensor_shape = input_tensor.shape
        return input_tensor.flatten()

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        return error_tensor.reshape(self.input_tensor_shape)
