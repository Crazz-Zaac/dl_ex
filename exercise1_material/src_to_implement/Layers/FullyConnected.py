import numpy as np
from Layers.Base import *
from Optimization.Optimizers import *


class FullyConnected(BaseLayer):

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

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        # print(f'input shape: {input_tensor.shape} weights shape: {self.weights.shape}')
        # print(np.hstack([input_tensor, np.ones((input_tensor.shape[0], 1))]).shape)
        input_tensor = np.hstack([input_tensor, np.ones((input_tensor.shape[0], 1))])
        self.input = input_tensor
        self.output_tensor = np.dot(input_tensor, self.weights) 
        # print(self.output_tensor.shape)
        
        return self.output_tensor

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        # computing the gradient tensor with respect to the weights
        self.error_tensor = np.dot(error_tensor, self.weights.T)
        
        # computing the gradient tensor with respect to the input tensor
        self._gradient_tensor = np.dot(np.atleast_2d(self.input).T, error_tensor)

        # print(f'gradient_tensor shape: {self.gradient_tensor.shape} error_tensor shape: {error_tensor.shape}')

        if self.optimizer is not None:
            # update weights using gradient with respect to weights
            # print(f'weights shape: {self.weights.shape} gradient shape: {self.gradient_tensor.shape}')
            self.weights = self.optimizer.calculate_update(
                self.weights, self._gradient_tensor
            )

        return self.error_tensor[:, :-1]

    @property
    def gradient_weights(self) -> np.ndarray:
        # do not perform an update if the optimizer is not set
        return self._gradient_tensor
