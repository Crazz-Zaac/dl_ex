import numpy as np
from Layers.Base import BaseLayer

class BatchNormalization(BaseLayer):
    """
    This class implements the batch normalization layer.
    It inherits from the BaseLayer class and implements the forward and backward pass.

    Attributes:
        input_size: int
            input size for the layer

        gamma: np.ndarray
            gamma for the layer
            shape: (input_size,)

        beta: np.ndarray
            beta for the layer
            shape: (input_size,)

        optimizer: object
            It is used to update the weights of the layer
            Formula: weights = weights - learning_rate * gradient

        gradient_tensor: np.ndarray
            It stores the gradient tensor with respect to the weights

        epsilon: float
            epsilon value to avoid division by zero

        running_mean: np.ndarray
            running mean for the layer
            shape: (input_size,)

        running_var: np.ndarray
            running variance for the layer
            shape: (input_size,)

        mean: np.ndarray
            mean for the layer
            shape: (batch_size, input_size)

        var: np.ndarray
            variance for the layer
            shape: (batch_size, input_size)

        normalized_input: np.ndarray
            normalized input for the layer
            shape: (batch_size, input_size)

        batch_size: int
            batch size for the layer

    """

    def __init__(self, channels: int, input_size: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.gamma = np.ones(input_size)  # weight gamma
        self.beta = np.zeros(input_size)  # bias beta
        self._optimizer = None
        self.gradient_tensor = None
        self.epsilon = epsilon
        self.running_mean = np.zeros(input_size)
        self.running_var = np.ones(input_size)
        self.mean = None
        self.var = None
        self.normalized_input = None
        self.batch_size = None

    # setter and getter property optimizer
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def initialize(self, weights_initializer, bias_initializer) -> None:
        """
        It initializes the weights of the batch normalization layer.

        Args:
            weights_initializer: str
                It is the initializer for the weights
                It can be 'random' or 'zeros'

            bias_initializer: str 
                It is the initializer for the bias
                It can be 'random' or 'zeros'
        """
        pass