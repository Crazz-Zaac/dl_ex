import numpy as np
from Layers.Base import BaseLayer
from Layers.Helpers import compute_bn_gradients


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

    def __init__(self, channels: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.channels = channels
        self.trainable = True

        self.initialize(None, None)

        self.alpha = 0.8
        self.epsilon = 1e-11  # Smaller than 1e-10

        self.test_mean = np.zeros(channels)
        self.test_var = np.ones(channels)

        self.mean = None
        self.var = None
        self.xhat = None

        self._gradient_bias = None
        self._gradient_weights = None
        self._optimizer = None
        self._bias_optimizer = None

    # setter and getter property optimizer and bias_optimizer
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def bias_optimizer(self):
        return self._bias_optimizer

    @bias_optimizer.setter
    def bias_optimizer(self, optimizer):
        self._bias_optimizer = optimizer

    # setter and getter property gradient_weights
    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights

    # setter and getter property gradient_bias
    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, gradient_bias):
        self._gradient_bias = gradient_bias

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
        # self.gamma = np.ones(self.channels)
        # self.beta = np.zeros(self.channels)

        self.gamma = weights_initializer.initialize(np.ones(self.channels))
        self.beta = bias_initializer.initialize(np.zeros(self.channels))

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        It performs the forward pass of the batch normalization layer.

        Args:
            input_tensor: np.ndarray
                input tensor for the layer
                shape: (batch_size, input_size)

        Returns:
            np.ndarray
                output tensor after forward pass
                shape: (batch_size, input_size)
        """
        self.input_tensor = input_tensor
        is_covolutional = len(input_tensor.shape) == 4
        self.is_covolutional = is_covolutional

        if not is_covolutional:
            mean_ax, var_ax = 0, 0
        else:
            mean_ax, var_ax = (0, 2, 3), (0, 2, 3)

        self.mean = np.mean(input_tensor, axis=mean_ax, keepdims=True)
        self.var = np.var(input_tensor, axis=var_ax, keepdims=True)

        if not is_covolutional:
            if self.testing_phase:
                self.x_hat = (input_tensor - self.test_mean) / np.sqrt(
                    self.test_var + self.epsilon
                )
            else:  # if it is not testing phase
                self._update_moving_statistics()  # update the moving statistics mean and variance

                # calculate the normalized input
                # x_hat = (x - mean) / sqrt(var + eps)
                self.x_hat = (input_tensor - self.mean) / np.sqrt(
                    self.test_var + self.epsilon
                )
            return self.gamma * self.x_hat + self.beta

        else:
            # if it is convolutional layer
            return self._conv_forward(input_tensor)

    def _update_moving_statistics(self):
        self.test_mean = self.alpha * self.test_mean + (1 - self.alpha) * self.mean
        self.test_var = self.alpha * self.test_var + (1 - self.alpha) * self.var

    def _conv_forward(self, input_tensor):
        # get the shape of the input tensor
        self.batch_size, channels, self.height, self.width = input_tensor.shape

        if self.testing_phase:
            # if it is testing phase normalize the input tensor
            # and return the normalized input tensor
            return (
                input_tensor - self.test_mean.reshape((1, channels, 1, 1))
            ) / np.sqrt(self.test_var.reshape((1, channels, 1, 1)) + self.epsilon)

        new_mean = np.mean(input_tensor, axis=(0, 2, 3), keepdims=True)
        new_var = np.var(input_tensor, axis=(0, 2, 3), keepdims=True)

        self.test_mean = np.mean(input_tensor, axis=(0, 2, 3), keepdims=True)
        self.test_var = np.var(input_tensor, axis=(0, 2, 3), keepdims=True)

        self.mean = new_mean
        self.var = new_var

        # calculate the normalized input tensor by using the formula
        # x_hat = (x - mean) / sqrt(var + eps)
        self.x_hat = (input_tensor - self.mean.reshape((1, channels, 1, 1))) / np.sqrt(
            self.var.reshape((1, channels, 1, 1)) + self.epsilon
        )

        # return the normalized input tensor
        return self.gamma.reshape((1, channels, 1, 1)) * self.x_hat + self.beta.reshape(
            (1, channels, 1, 1)
        )

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        """
        It performs the backward pass of the batch normalization layer.

        Args:
            error_tensor: np.ndarray
                error tensor for the layer
                shape: (batch_size, input_size)

        Returns:
            np.ndarray
                error tensor after backward pass
                shape: (batch_size, input_size)
        """
        if not self.is_covolutional:
            set_axis = 0
        else:
            set_axis = (0, 2, 3)

        if self.is_covolutional:
            error_calculated = compute_bn_gradients(
                self.reformat(error_tensor),
                self.reformat(self.input_tensor),
                self.gamma,
                self.mean,
                self.var,
                self.epsilon,
            )
            error_calculated = self.reformat(error_calculated)

        else:
            error_calculated = compute_bn_gradients(
                error_tensor, self.input_tensor, self.gamma, self.mean, self.var, self.epsilon
            )
        
        self.gradient_weights = np.sum(self.x_hat * error_tensor, axis=set_axis)
        self.gradient_bias = np.sum(error_tensor, axis=set_axis)
        
        if self.optimizer:
            self.gamma = self.optimizer.calculate_update(self.gamma, self.gradient_weights)
        
        if self.bias_optimizer:
            self.beta = self.bias_optimizer.calculate_update(self.beta, self.gradient_bias)
            

        return error_calculated

    def reformat(self, tensor: np.ndarray) -> np.ndarray:
        """
        It reformats the image-like tensor (with 4D shape) to a tensor with 2D shape
        and refortmats the vector-like tensor (with 2D shape) to a tensor with 4D shape.

        Args:
            tensor: np.ndarray
                tensor to be reformatted

        Returns:
            np.ndarray
                reformatted tensor
        """
        # if the tensor is 4D shape then reformat it to 2D shape
        if tensor.ndim == 4:
            batch, channels, height, width = tensor.shape
            reshaped = tensor.transpose(0, 2, 3, 1).reshape(-1, channels)
        else:
            batch, channels = tensor.shape
            side_length = int(np.sqrt(tensor.size / batch / channels))
            reshaped = tensor.reshape(batch, side_length, side_length, channels).transpose(0, 3, 1, 2)
        return reshaped
