import numpy as np

from Layers.Base import *
from Layers.Helpers import compute_bn_gradients


class BatchNormalization(BaseLayer):
    """
    This class implements the batch normalization layer.
    It inherits from the BaseLayer class and implements the forward and backward pass.

    Attributes:
        channels: int
            number of channels in the input tensor
        epsilon: float
            small value to avoid division by zero
        test_mean: np.ndarray
            mean of the input tensor during testing phase
        test_var: np.ndarray
            variance of the input tensor during testing phase
        x_tilde: np.ndarray
            normalized input tensor
        mean: np.ndarray
            mean of the input tensor
        var: np.ndarray
            variance of the input tensor
        _gradient_bias: np.ndarray
            gradient tensor with respect to the bias
        _gradient_weights: np.ndarray
            gradient tensor with respect to the weights
        _optimizer: object
            optimizer to update the weights
        _bias_optimizer: object
            optimizer to update the bias

    Methods:
        initialize:
            It initializes the weights and bias of the batch normalization layer.

        forward:
            Computes the forward pass for the batch normalization layer.

        backward:
            Computes the backward pass for the batch normalization layer.

        reformat:
            It reshapes the tensor to the required shape.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.trainable = True

        # initialize weights(gamma) and bias(beta)
        self.initialize(None, None)

        self.alpha = 0.8

        # for forward
        self.epsilon = 1e-11  # smaller than 1e-10

        # store running mean and variance
        self.test_mean = 0
        self.test_var = 1

        # intermediate values for forward and backward pass
        self.x_tilde = 0
        self.mean = 0
        self.var = 0
        self._gradient_bias = None
        self._gradient_weights = None

        # optimizer to update the weights and bias
        self._optimizer = None
        self._bias_optimizer = None

    #
    @property
    def bias_optimizer(self):
        return self._bias_optimizer

    @bias_optimizer.setter
    def bias_optimizer(self, bias_optimizer):
        self._bias_optimizer = bias_optimizer

    # property optimizer to get and set the optimizer
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    # property gradient_weights to get and set the gradient_weights during the backward pass
    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights

    # property gradient_bias to get and set the gradient_bias during the backward pass
    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, gradient_bias):
        self._gradient_bias = gradient_bias

    # initialize the weights and bias of the batch normalization layer
    def initialize(self, weights_initializer: object, bias_initializer: object) -> None:
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass for the batch normalization layer.
        The formula for batch normalization is:
        x_tilde = (x - mean) / sqrt(var + epsilon)
        y = weights * x_tilde + bias
        if it is not convolution:
            if testing_phase:
                x_tilde = (input_tensor - test_mean) / sqrt(test_var + epsilon)
            else:
                test_mean = alpha * mean + (1 - alpha) * mean
                test_var = alpha * var + (1 - alpha) * var
                x_tilde = (input_tensor - mean) / sqrt(var + epsilon)
        else:
            if testing_phase:
                y = (input_tensor - test_mean) / sqrt(test_var + epsilon)
            else:
                new_mean = mean(input_tensor)
                new_var = var(input_tensor)
                test_mean = alpha * mean + (1 - alpha) * new_mean
                test_var = alpha * var + (1 - alpha) * new_var
                x_tilde = (input_tensor - mean) / sqrt(var + epsilon)

        Args:
            input_tensor: np.ndarray
                The input tensor for the forward pass.

        Returns:
            np.ndarray
                The output tensor after applying the batch normalization layer.
        """
        # save input tensor for backward pass
        self.input_tensor = input_tensor

        # a flag to check if the input tensor is convolutional
        is_conv = len(input_tensor.shape) == 4
        self.is_conv = is_conv

        # determine along which axis to compute mean and variance
        # in case of convolution, mean and variance are computed along dimension(0), height(2) and width(3)
        mean_ax = 0 if not is_conv else (0, 2, 3)
        var_ax = 0 if not is_conv else (0, 2, 3)

        self.mean = np.mean(input_tensor, axis=mean_ax)
        self.var = np.var(input_tensor, axis=var_ax)

        if not self.is_conv:
            if self.testing_phase:  # normalizing the input tensor
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

            # normalize the input tensor using the computed mean and variance
            self.x_tilde = (
                self.input_tensor - self.mean.reshape((1, channels, 1, 1))
            ) / np.sqrt(self.var.reshape((1, channels, 1, 1)) + self.epsilon)

            # y = weights (gamma) * x_tilde + bias (beta)
            return self.weights.reshape(
                (1, channels, 1, 1)
            ) * self.x_tilde + self.bias.reshape((1, channels, 1, 1))

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        """
        Computes the backward pass for the batch normalization layer.
        If it is not convolution:
            if testing_phase:
                dy = error_tensor * weights
                dx_tilde = dy
                dvar = np.sum(dy * (input_tensor - test_mean) * (-0.5) * (test_var + epsilon) ** -1.5, axis=axis)
                dmean = np.sum(dy * (-1) / np.sqrt(test_var + epsilon), axis=axis) + dvar * np.sum(-2 * (input_tensor - test_mean), axis=axis) / batch
                dx = dx_tilde / np.sqrt(test_var + epsilon) + dvar * 2 * (input_tensor - test_mean) / batch + dmean / batch
                dgamma = np.sum(error_tensor * x_tilde, axis=axis)
                dbeta = np.sum(error_tensor, axis=axis)
            else:
                dy = error_tensor * weights
                dx_tilde = dy
                dvar = np.sum(dy * (input_tensor - mean) * (-0.5) * (var + epsilon) ** -1.5, axis=axis)
                dmean = np.sum(dy * (-1) / np.sqrt(var + epsilon), axis=axis) + dvar * np.sum(-2 * (input_tensor - mean), axis=axis) / batch
                dx = dx_tilde / np.sqrt(var + epsilon) + dvar * 2 * (input_tensor - mean) / batch + dmean / batch
                dgamma = np.sum(error_tensor * x_tilde, axis=axis)
                dbeta = np.sum(error_tensor, axis=axis)
        else:
            if testing_phase:
                dy = error_tensor
                dx_tilde = dy
                dvar = np.sum(dy * (input_tensor - test_mean) * (-0.5) * (test_var + epsilon) ** -1.5, axis=axis)
                dmean = np.sum(dy * (-1) / np.sqrt(test_var + epsilon), axis=axis) + dvar * np.sum(-2 * (input_tensor - test_mean), axis=axis) / batch
                dx = dx_tilde / np.sqrt(test_var + epsilon) + dvar * 2 * (input_tensor - test_mean) / batch + dmean / batch
                dgamma = np.sum(error_tensor * x_tilde, axis=axis)
                dbeta = np.sum(error_tensor, axis=axis)
            else:
                dy = error_tensor
                dx_tilde = dy
                dvar = np.sum(dy * (input_tensor - mean) * (-0.5) * (var + epsilon) ** -1.5, axis=axis)
                dmean = np.sum(dy * (-1) / np.sqrt(var + epsilon), axis=axis) + dvar * np.sum(-2 * (input_tensor - mean), axis=axis) / batch
                dx = dx_tilde / np.sqrt(var + epsilon) + dvar * 2 * (input_tensor - mean) / batch + dmean / batch
                dgamma = np.sum(error_tensor * x_tilde, axis=axis)
                dbeta = np.sum(error_tensor, axis=axis)

        if optimizer:
            update weights using gradient with respect to weights
        if bias_optimizer:
            update bias using gradient with respect to bias

        Args:
            error_tensor: np.ndarray
                The error tensor for the backward pass.

        Returns:
            np.ndarray
                The error tensor after applying the backward pass.
        """

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

        # calculate gradients for scale(gamma) and shift(beta) parameters using the error tensor and the normalized input tensor
        # here scale is the weights and shift is the bias
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
        """
        It reshapes the tensor to the required shape. If the input tensor is not convolutional, it reshapes the tensor
        to the required shape. If the input tensor is convolutional, it reshapes the tensor to the required shape.

        Args:
            tensor: np.ndarray
                The input tensor to be reshaped.

        Returns:
            np.ndarray
                The reshaped tensor.
        """

        if len(tensor.shape) == 4:
            batch, channels, height, width = tensor.shape
            reshaped_tensor = tensor.reshape((batch, channels, height * width))
            transposed_tensor = reshaped_tensor.transpose((0, 2, 1))
            output_tensor = transposed_tensor.reshape(
                (batch * height * width, channels)
            )
        else:
            batch, channels, height, width = self.input_tensor.shape
            reshaped_tensor = tensor.reshape((batch, height * width, channels))
            transposed_tensor = reshaped_tensor.transpose((0, 2, 1))
            output_tensor = transposed_tensor.reshape((batch, channels, height, width))

        return output_tensor
