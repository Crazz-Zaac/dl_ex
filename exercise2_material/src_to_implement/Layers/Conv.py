from Layers.Base import BaseLayer

import numpy as np
from scipy import signal
from copy import deepcopy as copy


class Conv(BaseLayer):
    """
    This class implements the convolutional layer.
    It inherits from the BaseLayer class and implements the forward and backward pass.

    Attributes:
        stride_shape: tuple or int
            The stride shape for the convolution operation.
            If it is an integer, it is converted to a tuple.

        convolution_shape: list
            The shape of the convolution operation.
            For 1D convolution, it is [channel, m].
            For 2D convolution, it is [channel, m, n].

        num_kernels: int
            The number of kernels for the convolution operation.

        weights: np.ndarray
            The weights for the convolution operation.
            Shape: (num_kernels, channel, m, n) for 2D convolution
                     (num_kernels, channel, m) for 1D convolution

        bias: np.ndarray
            The bias for the convolution operation.
            Shape: (num_kernels)

        gradient_weights: np.ndarray
            The gradient tensor with respect to the weights.

        gradient_bias: np.ndarray
            The gradient tensor with respect to the bias.

        optimizer: object
            The optimizer object to update the weights.

        bias_optimizer: object
            The optimizer object to update the bias.
    """

    def __init__(
        self, stride_shape: [tuple, int], convolution_shape: list, num_kernels: int
    ):
        super().__init__()
        self.trainable = True
        self.stride_shape = (
            (stride_shape[0], stride_shape[0])
            if len(stride_shape) == 1
            else stride_shape
        )
        # convolution shape: [channel, m] for 1D convolution, [channel, m, n] for 2D convolution
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        # weight shape: (num_kernels, channel, m, n) for 2D convolution
        # (num_kernels, channel, m) for 1D convolution
        # bias shape: (num_kernels)
        self._weight_shape = [num_kernels, *convolution_shape]
        self._bias_shape = [num_kernels]

        # initialize weights and bias with random values between 0 and 1 using uniform distribution
        self.weights = np.random.uniform(0, 1, self._weight_shape)
        self.bias = np.random.rand(self._bias_shape[0])

        # gradient parameters
        self._grad_wts = None
        self._grad_bias = None

        self._optm = None
        self._bias_optm = None

        # check if the convolution shape is 1D or 2D
        self.conv_dimension = 2 if len(convolution_shape) == 3 else 1

    def initialize(self, weights_initializer: object, bias_initializer: object) -> None:
        """
        This method initializes the weights and bias of the convolutional layer.

        Args:
            weights_initializer: object
                The initializer object for the weights.

            bias_initializer: object
                The initializer object for the bias.
        """
        fa_in = 1  # number of input features
        fa_out = self.num_kernels  # number of output features

        # all the dimensions of the convolution shape are multiplied to get the number of input features
        # except the first dimension of the convolution shape is multiplied to get the number of output features
        for dim_val in self.convolution_shape:
            fa_in *= dim_val
        for dim_val in self.convolution_shape[1:]:
            fa_out *= dim_val

        self.weights = weights_initializer.initialize(self._weight_shape, fa_in, fa_out)
        self.bias = bias_initializer.initialize(self._bias_shape, 1, self.num_kernels)

        self._optm = copy(self.optimizer)
        self._bias_optm = copy(self.optimizer)

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        This method performs the forward pass through the convolutional layer.
        It accepts an input tensor, performs the convolution operation on the input tensor
        with the weights and bias, and returns the output tensor.

        Args:
            input_tensor: np.ndarray
                The input tensor for the forward pass.
                Size: (batch_size, channels, height, width) for 2D convolution
                        (batch_size, channels, length) for 1D convolution

        Returns:
            np.ndarray
                The output tensor after applying the forward pass.
                Size: (batch_size, num_kernels, height, width) for 2D convolution
                        (batch_size, num_kernels, length) for 1D convolution

        """
        self.inp_tensor = input_tensor
        ishape = input_tensor.shape
        self.inp_shape = ishape
        batch_size, channels, y, x = (
            ishape if self.conv_dimension == 2 else (*ishape, None)
        )
        # get the last two dimensions of the convolution shape for the kernel size
        # for 2d conv: (m, n), for 1d conv: (c, m)
        kernel_width, kernel_height = self.convolution_shape[-2:]

        stride_w, stride_h = self.stride_shape

        # padding to keep the output shape same as input shape
        pad = [(kernel_width - 1) / 2]
        output_shape = [int((y - kernel_width + 2 * pad[0]) / stride_w + 1)]
        # if 2d convolution, pad the height dimension
        if self.conv_dimension == 2:
            pad.append((kernel_height - 1) / 2)
            output_shape.append(int((x - kernel_height + 2 * pad[1]) / stride_h + 1))

        self.pad = pad
        # initialize the output tensor with zeros of the output shape and batch size
        result = np.zeros((batch_size, self.num_kernels, *output_shape))

        # iterate over the batch size, number of kernels, and channels of the input tensor
        for current_batch in range(batch_size):
            for current_kernel in range(self.num_kernels):
                # initialize the correlation output with zeros
                # if x is None, it is 1D convolution with shape (y) else 2D convolution with shape (y, x)
                corr_out = np.zeros((y, x)) if x else np.zeros((y))

                # iterate over the channels of the input tensor and the weights
                # using the correlation operation instead of convolution operation for forward pass
                for channel in range(channels):
                    # correlation operation between the input tensor and the weights
                    corr_out += signal.correlate(
                        input_tensor[current_batch, channel],
                        self.weights[current_kernel, channel],
                        mode="same",
                        method="direct",
                    )

                # stride the correlation output to match the output shape
                # if 2D convolution, stride the correlation output in both dimensions else in one dimension
                corr_out = (
                    corr_out[::stride_w, ::stride_h]
                    if self.conv_dimension == 2
                    else corr_out[::stride_w]
                )

                # finally add the bias term to the correlation output and store
                # the result in the corresponding position of the output tensor
                result[current_batch, current_kernel] = (
                    corr_out + self.bias[current_kernel]
                )

        return result

    def update_parameters(self, error_tensor: np.ndarray) -> None:
        """
        It updates the weights and bias of the layer.

        Args:
            error_tensor: np.ndarray
                The error tensor of the current layer.

        Returns:
            None
        """

        batch_error_h = error_tensor.sum(axis=0).sum(axis=1)

        # sum the error tensor over the batch size and the output shape 
        # if 2D convolution, sum over the height and width dimensions else over the length dimension
        self._grad_bias = (
            batch_error_h.sum(axis=1) if self.conv_dimension == 2 else batch_error_h
        )

        # compute gradients of weights for each kernel and channel of input
        batch_size, channels, y, x = (
            self.inp_shape if self.conv_dimension == 2 else (*self.inp_shape, None)
        )

        stride_w, stride_h = self.stride_shape
        kernel_width, kernel_height = self.convolution_shape[-2:]

        # initialize gradient weights with zeros
        self.gradient_weights = np.zeros_like(self.weights)
        # if used correlation in forward, should use convolve in backward
        # or vice versa because of the sign change in the formula
        for current_batch in range(batch_size):
            for channel in range(channels):
                for current_kernel in range(self.num_kernels):

                    if self.conv_dimension == 2:  # 2D convolution
                        error = np.zeros((y, x))
                        # stride the error tensor to match the input tensor
                        error[::stride_w, ::stride_h] = error_tensor[
                            current_batch, current_kernel
                        ]
                        # pad the input tensor for convolution operation with the error tensor
                        # to calculate the gradient of weights
                        padded_input = np.pad(
                            self.inp_tensor[current_batch, channel],
                            [
                                (int(np.ceil(self.pad[0])), int(np.floor(self.pad[0]))),
                                (int(np.ceil(self.pad[1])), int(np.floor(self.pad[1]))),
                            ],
                        )
                    else:  # 1D convolution
                        error = np.zeros(y)
                        # stride the error tensor to match the input tensor
                        error[::stride_w] = error_tensor[current_batch, current_kernel]
                        padded_input = np.pad(
                            self.inp_tensor[current_batch, channel],
                            [(int(np.ceil(self.pad[0])), int(np.floor(self.pad[0])))],
                        )

                    self.gradient_weights[current_kernel, channel] += signal.correlate(
                        padded_input, error, mode="valid"
                    )

        # update weights and bias using optimizer object
        if self.optimizer:
            self.weights = self._optm.calculate_update(self.weights, self._grad_wts)
            self.bias = self._bias_optm.calculate_update(self.bias, self._grad_bias)

    def error_this_layer(self, error_tensor: np.ndarray) -> np.ndarray:
        """
        It calculates the error tensor of the current layer.

        Args:
            error_tensor: np.ndarray
                The error tensor of the next layer.

        Returns:
            np.ndarray
                The error tensor of the current layer.
        """

        gradient = np.zeros_like(self.inp_tensor)
        stride_w, stride_h = self.stride_shape
        # new_weight shape: (num_kernels, channel, m, n)
        # new weight is the transposed version of the current weight tensor
        # for 2d conv: (num_kernels, channel, m, n) -> (channel, num_kernels, m, n)
        # for 1d conv: (num_kernels, channel, m) -> (channel, num_kernels, m)
        new_weight = self.weights.copy()
        new_weight = (
            np.transpose(new_weight, axes=(1, 0, 2, 3))
            if self.conv_dimension == 2
            else np.transpose(new_weight, axes=(1, 0, 2))
        )
        # get the shape of the input tensor after the convolution operation
        # h_inpt and w_inpt are the height and width of the input tensor
        new_input_shape = self.inp_tensor.shape
        h_inpt, w_inpt = (
            new_input_shape[-2:]
            if self.conv_dimension == 2
            else (new_input_shape[-1], None)
        )

        batch_size = self.inp_tensor.shape[0]

        # w_kernel and w_channel are the number of kernels and channels of the weight tensor
        w_kernel, w_channel = new_weight.shape[:2]

        for current_batch in range(batch_size):
            for current_kernel in range(w_kernel):
                new_gradient = 0
                # iterate over the channels of the weight tensor
                # if 2D convolution, stride the error tensor in both dimensions else in one dimension
                for current_channel in range(w_channel):
                    if self.conv_dimension == 2:  # 2D convolution
                        error = np.zeros((h_inpt, w_inpt))
                        error[::stride_w, ::stride_h] = error_tensor[
                            current_batch, current_channel
                        ]
                    else:
                        error = np.zeros(h_inpt)  # 1D convolution
                        error[::stride_w] = error_tensor[current_batch, current_kernel]

                    # convolution operation to calculate the gradient of the input tensor
                    # between the error tensor and the transposed weight tensor
                    new_gradient += signal.convolve(
                        error,
                        new_weight[current_kernel, current_channel],
                        mode="same",
                        method="direct",
                    )

                gradient[current_batch, current_kernel] = new_gradient
        return gradient

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        """
        This method performs the backward pass through the convolutional layer.
        It accepts the error tensor of the next layer, calculates the error tensor of the current layer,
        and updates the weights and bias.

        Args:
            error_tensor: np.ndarray
                The error tensor of the next layer.

        Returns:
            np.ndarray
                The error tensor of the current layer.
        """
        self.update_parameters(error_tensor)
        # calculate the error tensor of the current layer
        gradient = self.error_this_layer(error_tensor)

        return gradient

    # getter and setter methods for the attributes of the class
    # gradient_weights -> getter and setter for the gradient_weights attribute
    # gradient_bias -> getter and setter for the gradient_bias attribute
    # optimizer -> getter and setter for the optimizer attribute
    # bias_optimizer -> getter and setter for the bias_optimizer attribute
    @property
    def gradient_weights(self):
        return self._grad_wts

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._grad_wts = value

    @property
    def gradient_bias(self):
        return self._grad_bias

    @gradient_bias.setter
    def gradient_bias(self, value):
        self._grad_bias = value

    @property
    def optimizer(self):
        return self._optm

    @optimizer.setter
    def optimizer(self, value):
        self._optm = value

    @property
    def bias_optimizer(self):
        return self._bias_optm

    @bias_optimizer.setter
    def bias_optimizer(self, value):
        self._bias_optm = value
