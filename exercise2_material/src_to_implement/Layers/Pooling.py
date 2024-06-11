import numpy as np


class Pooling:
    def __init__(self, pooling_shape: tuple, stride_shape: tuple) -> None:
        """
        Initializes the pooling layer.

        Args:
            pool_shape: tuple
                The shape of the pooling operation is [m, n] for 2D pooling.
                m = height of the pooling window, n = width of the pooling window.
            stride_shape: tuple
                The stride shape for the pooling operation.
        """
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
        batch_size, channels, input_height, input_width = input_tensor.shape
        kernel_height, kernel_width = self.pooling_shape
        stride_h, stride_w = self.stride_shape
        output_height = (input_height - kernel_height) // stride_h + 1
        output_width = (input_width - kernel_width) // stride_w + 1
        # using valid padding for pooling layer
        output_tensor = np.zeros((batch_size, channels, output_height, output_width))
        self.mask = np.zeros(input_tensor.shape)

        for batch in range(batch_size):
            for channel in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        output_tensor[batch, channel, i, j] = np.max(
                            input_tensor[
                                batch,
                                channel,
                                i * stride_h : i * stride_h + kernel_height,
                                j * stride_w : j * stride_w + kernel_width,
                            ]
                        )
                        max_indices = np.unravel_index(
                            np.argmax(
                                input_tensor[
                                    batch,
                                    channel,
                                    i * stride_h : i * stride_h + kernel_height,
                                    j * stride_w : j * stride_w + kernel_width,
                                ]
                            ),
                            (kernel_height, kernel_width),
                        )
                        self.mask[
                            batch,
                            channel,
                            i * stride_h + max_indices[0],
                            j * stride_w + max_indices[1],
                        ] = 1
                        
        return output_tensor

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Performs the backward pass of the pooling layer.

        Args:
            grad: np.ndarray
                The gradient tensor of shape [batch_size, channels, height, width] for 2D pooling
                and [batch_size, channels, length] for 1D pooling.

        Returns:
            np.ndarray
                The gradient tensor of shape [batch_size, channels, height, width] for 2D pooling
                and [batch_size, channels, length] for 1D pooling.
        """
        pass
