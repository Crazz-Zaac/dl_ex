from Layers.Base import BaseLayer

import numpy as np
from scipy import signal
from copy import deepcopy as copy


class Conv(BaseLayer):
    '''
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
    '''
    
    def __init__(self, stride_shape: [tuple, int], convolution_shape: list, num_kernels: int):
        super().__init__()
        self.trainable = True
        self.stride_shape = (
            (stride_shape[0], stride_shape[0])
            if len(stride_shape) == 1
            else stride_shape
        )
        # 1d as [channel,m], 2d as [channel,m,n]
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        # init weights as uniform random (will be initialized again with initialize method)
        # shape for 2d conv: (num_kernels, channel, m, n)
        self._weight_shape = [num_kernels, *convolution_shape]
        self._bias_shape = [num_kernels]
        self.weights = np.random.uniform(0, 1, self._weight_shape)
        # # bias shape: number of kernels
        self.bias = np.random.rand(self._bias_shape[0])

        # grad parameters
        self._grad_wts = None
        self._grad_bias = None

        self._optm = None
        self._bias_optm = None

        # conv_dimension if it is 2d or 1d
        self.conv_dimension = 2 if len(convolution_shape) == 3 else 1

    def initialize(self, weights_initializer: object, bias_initializer: object) -> None:
        '''
        This method initializes the weights and bias of the convolutional layer.
        
        Args:
            weights_initializer: object
                The initializer object for the weights.
                
            bias_initializer: object
                The initializer object for the bias.        
        '''
        
        fa_in = 1
        for i in self.convolution_shape:
            fa_in *= i
        fa_out = self.num_kernels
        for i in self.convolution_shape[1:]:
            fa_out *= i
        self.weights = weights_initializer.initialize(self._weight_shape, fa_in, fa_out)

        self.bias = bias_initializer.initialize(self._bias_shape, 1, self.num_kernels)

        self._optm = copy(self.optimizer)
        self._bias_optm = copy(self.optimizer)

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        '''
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
                        
        '''
        
        # if correlation is used in forward, we can use convole in backward
        # or vice versa
        # input_tensor shape (b,c,x,y) or (b,c,x)
        self.inp_tensor = input_tensor
        ishape = input_tensor.shape
        self.inp_shape = ishape
        batch_size, channels, y, x = (
            ishape if self.conv_dimension == 2 else (*ishape, None)
        )
        cw, ch = self.convolution_shape[-2:]

        stride_w, stride_h = self.stride_shape

        # new shape of y = (y-ky + 2*p)/stride_w + 1; y input size, ky kernel size, p padding size, stride_w stride size
        #  but we need o/p size same as i/p so p=(ky-1)/2 if stride_w==1
        # else we need to derive
        pad = [(cw - 1) / 2]
        output_shape = [int((y - cw + 2 * pad[0]) / stride_w + 1)]
        if self.conv_dimension == 2:
            pad.append((ch - 1) / 2)
            output_shape.append(int((x - ch + 2 * pad[1]) / stride_h + 1))
        self.pad = pad
        result = np.zeros((batch_size, self.num_kernels, *output_shape))

        # if used correlation in forward, should use convolve in backward
        for current_batch in range(batch_size):
            for current_kernel in range(self.num_kernels):
                # sum outputs of correlation of this kernel with individual input channel of input
                kout = np.zeros((y, x)) if x else np.zeros((y))
                for ch in range(channels):
                    # correlate with this batch's this channel and this kernel's this channel
                    kout += signal.correlate(
                        input_tensor[current_batch, ch],
                        self.weights[current_kernel, ch],
                        mode="same",
                        method="direct",
                    )

                kout = (
                    kout[::stride_w, ::stride_h]
                    if self.conv_dimension == 2
                    else kout[::stride_w]
                )
                result[current_batch, current_kernel] = kout + self.bias[current_kernel]

        return result

    def update_parameters(self, error_tensor: np.ndarray) -> None:
        '''
        It updates the weights and bias of the layer.
        
        Args:  
            error_tensor: np.ndarray
                The error tensor of the current layer.
                
        Returns:
            None
        '''
        
        # compute gradients of weights and bias
        berror = error_tensor.sum(axis=0)
        # sum over all batches
        yerror = berror.sum(axis=1)
        # sum over all channels
        self._grad_bias = yerror.sum(axis=1) if self.conv_dimension == 2 else yerror

        # compute gradients of weights for each kernel and channel of input
        batch_size, channels, y, x = (
            self.inp_shape if self.conv_dimension == 2 else (*self.inp_shape, None)
        )

        stride_w, stride_h = self.stride_shape
        cw, ch = self.convolution_shape[-2:]

        # initialize gradient weights with zeros
        self.gradient_weights = np.zeros_like(self.weights)
        # if used correlation in forward, should use convolve in backward
        # or vice versa because of the sign change in the formula
        for current_batch in range(batch_size):
            for ch in range(channels):
                for current_kernel in range(self.num_kernels):

                    if self.conv_dimension == 2:
                        error = np.zeros((y, x))
                        error[::stride_w, ::stride_h] = error_tensor[
                            current_batch, current_kernel
                        ]
                        padded_input = np.pad(
                            self.inp_tensor[current_batch, ch],
                            [
                                (int(np.ceil(self.pad[0])), int(np.floor(self.pad[0]))),
                                (int(np.ceil(self.pad[1])), int(np.floor(self.pad[1]))),
                            ],
                        )
                    else:
                        error = np.zeros(y)
                        error[::stride_w] = error_tensor[current_batch, current_kernel]
                        padded_input = np.pad(
                            self.inp_tensor[current_batch, ch],
                            [(int(np.ceil(self.pad[0])), int(np.floor(self.pad[0])))],
                        )

                    self.gradient_weights[current_kernel, ch] += signal.correlate(
                        padded_input, error, mode="valid"
                    )

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

        nweight = self.weights.copy()
        nweight = (
            np.transpose(nweight, axes=(1, 0, 2, 3))
            if self.conv_dimension == 2
            else np.transpose(nweight, axes=(1, 0, 2))
        )
        ishape = self.inp_tensor.shape
        y, x = ishape[-2:] if self.conv_dimension == 2 else (ishape[-1], None)

        batch_size = self.inp_tensor.shape[0]
        wk, wc = nweight.shape[:2]

        for current_batch in range(batch_size):
            for ck in range(wk):
                grad = 0
                for c in range(wc):
                    if self.conv_dimension == 2:
                        err = np.zeros((y, x))
                        err[::stride_w, ::stride_h] = error_tensor[current_batch, c]
                    else:
                        err = np.zeros(y)
                        err[::stride_w] = error_tensor[current_batch, ck]
                    # we used correlate on forward, use convolve now
                    grad += signal.convolve(
                        err, nweight[ck, c], mode="same", method="direct"
                    )

                gradient[current_batch, ck] = grad
        return gradient

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        self.update_parameters(error_tensor)
        gradient = self.error_this_layer(error_tensor)

        return gradient

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
