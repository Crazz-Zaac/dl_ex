import numpy as np


class Constant:
    def __init__(self, value: float = 0.1) -> None:
        self.value = value

    def initialize(self, weights_shape, fan_in, fan_out) -> np.ndarray:
        """
        Initialize the weights with a constant value and return the weights.

        Args:
        - weights_shape: tuple of integers, shape of the weight matrix

        - fan_in: int, dimension of the input
            for convolutional layers: number of input channels * kernel height * kernel width
            for fully connected layers: number of input features

        - fan_out: int, dimension of the output
            for convolutional layers: number of filters * kernel height * kernel width
            for fully connected layers: number of output features

        Returns:
        - weights: np.ndarray, shape = weight_shape
        """
        
        tensor = np.full(weights_shape, self.value)
        
        return tensor


class UniformRandom:
    def __init__(self) -> None:
        pass

    def initialize(self, weight_shape: np.ndarray, fan_in: int, fan_out: int) -> np.ndarray:
        """
        Initialize the weights with a uniform random distribution and return the weights.

        Args:
        - weight_shape: tuple of integers, shape of the weight matrix

        - fan_in: int, dimension of the input
            for convolutional layers: number of input channels * kernel height * kernel width
            for fully connected layers: number of input features

        - fan_out: int, dimension of the output
            for convolutional layers: number of filters * kernel height * kernel width
            for fully connected layers: number of output features

        Returns:
        - weights: np.ndarray, shape = weight_shape
        """
        
        tensor = np.random.uniform(0, 1, weights_shape)
        
        return tensor



class Xavier:
    def __init__(self) -> None:
        pass

    def initialize(self, weight_shape: np.ndarray, fan_in: int, fan_out: int) -> np.ndarray:
        """
        Initialize the weights with the Xavier initialization and return the weights.

        Args:
        - weight_shape: tuple of integers, shape of the weight matrix

        - fan_in: int, dimension of the input
            for convolutional layers: number of input channels * kernel height * kernel width
            for fully connected layers: number of input features

        - fan_out: int, dimension of the output
            for convolutional layers: number of filters * kernel height * kernel width
            for fully connected layers: number of output features

        Returns:
        - weights: np.ndarray, shape = weight_shape
        """
        sigma = np.sqrt(2/(fan_in + fan_out))
        
        tensor = np.random.normal(0, sigma, weights_shape)
        
        return tensor
    
class He:
    def __init__(self) -> None:
        self.weight_shape = None

    def initialize(self, weight_shape: np.ndarray, fan_in: int, fan_out: int) -> np.ndarray:
        """
        Initialize the weights with the He initialization and return the weights.

        Args:
        - weight_shape: tuple of integers, shape of the weight matrix

        - fan_in: int, dimension of the input
            for convolutional layers: number of input channels * kernel height * kernel width
            for fully connected layers: number of input features

        - fan_out: int, dimension of the output
            for convolutional layers: number of filters * kernel height * kernel width
            for fully connected layers: number of output features

        Returns:
        - weights: np.ndarray, shape = weight_shape
        """
        
        sigma = np.sqrt(2/fan_in)
        
        tensor = np.random.normal(0, sigma, weights_shape)
        
        return tensor  # shape = weight_shape