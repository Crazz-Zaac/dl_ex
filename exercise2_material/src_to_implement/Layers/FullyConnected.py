import numpy as np
from Layers.Base import BaseLayer
from Optimization.Optimizers import *
from Layers.Initializers import Constant, UniformRandom, Xavier, He


class FullyConnected(BaseLayer):
    """
    This class implements the fully connected layer.
    It inherits from the BaseLayer class and implements the forward and backward pass.

    Attributes:
        input_size: int
            input size for the layer

        output_size: int
            output size for the layer

        weights: np.ndarray
            weights for the layer
            shape: (input_size + 1, output_size) - The +1 is for the bias term

        optimizer: object
            It is used to update the weights of the layer
            Formula: weights = weights - learning_rate * gradient

        gradient_tensor: np.ndarray
            It stores the gradient tensor with respect to the weights

    """

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(size=(input_size + 1, output_size))
        self._optimizer = None
        self.gradient_tensor = None

    # setter and getter property optimizer
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def initialize(self, weights_initializer, bias_initializer) -> None:
        """
        It initializes the weights of the fully connected layer.

        Args:
            weights_initializer: str
                It is the initializer for the weights
                It can be 'random' or 'zeros'

            bias_initializer: str
                It is the initializer for the bias
                It can be 'random' or 'zeros'
        """
        weights_shape = (self.input_size, self.output_size)
        bias_shape = (1, self.output_size)

        weights = weights_initializer.initialize(
            weights_shape, self.input_size, self.output_size
        )
        bias = bias_initializer.initialize(
            bias_shape, 1, self.output_size
        )
        # the vstack function adds the bias term to the weights matrix
        self.weights = np.vstack([weights, bias])

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        It performs the forward pass through the fully connected layer.
        This method accepts an input tensor, adds a bias term to the input tensor,
        and computes the dot product (matrix multiplication) of the input tensor and the weights.

        Args:
            input_tensor: np.ndarray
                It is the input tensor for the forward pass

        Returns:
            np.ndarray
                output tensor after applying the forward pass
        """

        # print(f'input shape: {input_tensor.shape} weights shape: {self.weights.shape}')
        # print(np.hstack([input_tensor, np.ones((input_tensor.shape[0], 1))]).shape)
        input_tensor = np.hstack([input_tensor, np.ones((input_tensor.shape[0], 1))])
        self.input = input_tensor
        self.output_tensor = np.dot(input_tensor, self.weights)
        # print(self.output_tensor.shape)

        return self.output_tensor

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        """
        It performs the backward pass through the fully connected layer.
        This method accepts an error tensor, computes the error tensor with respect to the input tensor,
        updates the weights using the optimizer, and computes the gradient tensor with respect to the weights.

        Args:
            error_tensor: np.ndarray
                error tensor for the backward pass
                shape:

        Returns:
            np.ndarray
                error tensor after applying the backward pass
        """

        print(f"Beginning Error tensor shape: {error_tensor.shape}")
        # computing the error tensor with respect to the input tensor
        self.error_tensor = np.dot(error_tensor, self.weights.T)

        # computing the gradient tensor with respect to the input tensor
        self._gradient_tensor = np.dot(np.atleast_2d(self.input).T, error_tensor)

        if self.optimizer is not None:
            # update weights using gradient with respect to weights
            # print(f'weights shape: {self.weights.shape} gradient shape: {self.gradient_tensor.shape}')
            self.weights = self.optimizer.calculate_update(
                self.weights, self._gradient_tensor
            )
        print(f"Error tensor shape: {self.error_tensor.shape}")
        print(f"Final error_tensor shape: {self.error_tensor[:, :-1].shape}\n")
        return self.error_tensor[:, :-1]

    @property
    def gradient_weights(self) -> np.ndarray:
        """
        This method returns the gradient tensor with respect to the weights.
        """
        return self._gradient_tensor
