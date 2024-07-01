import numpy as np

from Layers.Base import *
from Layers.Helpers import compute_bn_gradients


class BatchNormalization(BaseLayer):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.trainable = True

        # initialize weights(gamma) and bias(beta)
        self.initialize(None, None)

        self.alpha = 0.8

        # for forward
        self.epsilon = 1e-11  # smaller than 1e-10
        self.test_mean = 0
        self.test_var = 1
        self.x_tilde = 0
        self.mean = 0
        self.var = 0

        # for backward
        self._gradient_bias = None
        self._gradient_weights = None
        self._optimizer = None
        self._bias_optimizer = None

    @property
    def bias_optimizer(self):
        return self._bias_optimizer

    @bias_optimizer.setter
    def bias_optimizer(self, opt):
        self._bias_optimizer = opt

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, x):
        self._gradient_weights = x

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, x):
        self._gradient_bias = x

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)


    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        self.input_tensor = input_tensor

        is_conv = len(input_tensor.shape) == 4
        self.is_conv = is_conv

        mean_ax = 0 if not is_conv else (0, 2, 3)
        var_ax = 0 if not is_conv else (0, 2, 3)

        self.mean = np.mean(input_tensor, axis=mean_ax)
        self.var = np.var(input_tensor, axis=var_ax)

        if not self.is_conv:
            if self.testing_phase:
                self.x_tilde = (input_tensor - self.test_mean) / np.sqrt(
                    self.test_var + self.epsilon
                )
            else:
                self.test_mean = self.alpha * self.mean + (1 - self.alpha) * self.mean
                self.test_var = self.alpha * self.var + (1 - self.alpha) * self.var

                self.x_tilde = (self.input_tensor - self.mean) / np.sqrt(
                    self.var + self.epsilon
                )
            return self.weights * self.x_tilde + self.bias
        else:
            bsize, channels, *_ = input_tensor.shape
            if self.testing_phase:
                return (
                    self.input_tensor - self.test_mean.reshape((1, channels, 1, 1))
                ) / (self.test_var.reshape((1, channels, 1, 1)) + self.epsilon) ** 0.5

            new_mean = np.mean(self.input_tensor, axis=mean_ax)
            new_var = np.var(self.input_tensor, axis=var_ax)

            self.test_mean = self.alpha * self.mean.reshape((1, channels, 1, 1)) + (
                1 - self.alpha
            ) * new_mean.reshape((1, channels, 1, 1))
            self.test_var = self.alpha * self.var.reshape((1, channels, 1, 1)) + (
                1 - self.alpha
            ) * new_var.reshape((1, channels, 1, 1))

            self.mean = new_mean
            self.var = new_var

            self.x_tilde = (
                self.input_tensor - self.mean.reshape((1, channels, 1, 1))
            ) / np.sqrt(self.var.reshape((1, channels, 1, 1)) + self.epsilon)

            return self.weights.reshape(
                (1, channels, 1, 1)
            ) * self.x_tilde + self.bias.reshape((1, channels, 1, 1))

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:

        axis = 0 if not self.is_conv else (0, 2, 3)

        if self.is_conv:
            err_here = compute_bn_gradients(
                self.reformat(error_tensor),
                self.reformat(self.input_tensor),
                self.weights,
                self.mean,
                self.var,
                self.epsilon,
            )
            err_here = self.reformat(err_here)

        else:
            err_here = compute_bn_gradients(
                error_tensor,
                self.input_tensor,
                self.weights,
                self.mean,
                self.var,
                self.epsilon,
            )

        self.gradient_weights = np.sum(error_tensor * self.x_tilde, axis=axis)
        self.gradient_bias = np.sum(error_tensor, axis=axis)

        if self.optimizer:
            self.weights = self.optimizer.calculate_update(
                self.weights, self._gradient_weights
            )
        if self.bias_optimizer:
            self.bias = self.bias_optimizer.calculate_update(
                self.bias, self._gradient_bias
            )

        return err_here

    def reformat(self, tensor: np.ndarray) -> np.ndarray:
        if len(tensor.shape) == 4:
            batch, channels, height, width = tensor.shape
            
            
            

        return reshaped_tensor
