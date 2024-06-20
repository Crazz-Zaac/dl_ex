import numpy as np
from Layers.Base import BaseLayer


class Pooling(BaseLayer):
    def __init__(self, stride_shape: tuple, pooling_shape: tuple) -> None:
        """
        Initializes the pooling layer.

        Args:
            pool_shape: tuple
                The shape of the pooling operation is [m, n] for 2D pooling.
                m = height of the pooling window, n = width of the pooling window.
            stride_shape: tuple
                The stride shape for the pooling operation.
        """
        super().__init__()
        self.pooling_shape = pooling_shape
        self.stride_shape = stride_shape

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of the pooling layer.

        Args:
            X: np.ndarray
                The input tensor of shape [batch_size, channels, height, width] for 2D pooling
                and [batch_size, channels, length] for 1D pooling.

        Returns:
            np.ndarray
                The output tensor of shape [batch_size, channels, height, width] for 2D pooling
                and [batch_size, channels, length] for 1D pooling.
        """
        # save the input tensor for the backward pass
        self.input_tensor = input_tensor
        batch_size, channels, input_height, input_width = input_tensor.shape
        
        pooling_h, pooling_w = self.pooling_shape
        stride_h, stride_w = self.stride_shape
        
        output_height = int(1 + (input_height - pooling_h) / stride_h)
        output_width = int(1 + (input_width - pooling_w) / stride_w)
        output_tensor = np.zeros((batch_size, channels, output_height, output_width))

        for batch in range(batch_size):
            for channel in range(channels):
                for h_out in range(output_height):
                    for w_out in range(output_width):
                        # We take the max value in the input patch and assign it to the output tensor
                        output_tensor[batch, channel, h_out, w_out] = np.max(
                            input_tensor[
                                batch,
                                channel,
                                h_out * stride_h : h_out * stride_h + pooling_h,
                                w_out * stride_w : w_out * stride_w + pooling_w,
                            ]
                        )
        self.output_tensor = output_tensor
        return output_tensor

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        '''
        This method performs the backward pass of the pooling layer. 
        It calculates the gradient of the loss with respect to the input tensor.
        
        Args:
            error_tensor: np.ndarray
                The error tensor of shape [batch_size, channels, height, width] for 2D pooling
                and [batch_size, channels, length] for 1D pooling.
            
        Returns:
            np.ndarray
                The gradient of the loss with respect to the input tensor. 
                The shape of the returned tensor is the same as the shape of the input tensor.
        '''
        # initialize the gradient tensor with zeros of the same shape as the input tensor
        gradient = np.zeros_like(self.input_tensor)
        batch_size, channels, input_height, input_width = self.input_tensor.shape
        
        pooling_h, pooling_w = self.pooling_shape
        stride_h, stride_w = self.stride_shape
        
        output_height, output_width = self.output_tensor.shape[-2:]

        for batch in range(batch_size):
            for channel in range(channels):
                for h_out in range(output_height):
                    for w_out in range(output_width):
                        # first we find the position of the max value in the input patch
                        input_patch = self.input_tensor[
                            batch,
                            channel,
                            h_out * stride_h : h_out * stride_h + pooling_h,
                            w_out * stride_w : w_out * stride_w + pooling_w,
                        ]
                        # find the position of the max value in the input patch
                        max_val = self.output_tensor[batch, channel, h_out, w_out]
                        x_max, y_max = np.where(input_patch == max_val)
                        
                        # if there are multiple max values, we take the first one
                        x_max, y_max = x_max[0], y_max[0]
                        
                        # finally, we add the error tensor to the gradient tensor
                        gradient[
                            batch,
                            channel,
                            h_out * stride_h + x_max,
                            w_out * stride_w + y_max,
                        ] += error_tensor[batch, channel, h_out, w_out]

        return gradient
