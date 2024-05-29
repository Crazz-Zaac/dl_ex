import numpy as np

# Basic optimizer that performs a single step of gradient descent
# on the given weight tensor.
class Sgd:
    '''
    Performs a single step of gradient descent on the given weight tensor.
    
    Args:
        learning_rate: The learning rate to use.
    
    Returns:
        The updated weight tensor.
    '''
    
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate

    def calculate_update(
        self, weight_tensor: np.ndarray, gradient_tensor: np.ndarray
    ) -> np.ndarray:
        return weight_tensor - self.learning_rate * gradient_tensor

