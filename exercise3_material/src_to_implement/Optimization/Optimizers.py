import numpy as np
from Optimization.Constraints import L2_Regularizer, L1_Regularizer

#creating a base class for all the optimizers and defining the common methods
class Optimizer:
    """
    Base class for all optimizers.

    Args:
        learning_rate: The learning rate to use.
    """

    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate
        self.regularizer = None
    
    def add_regularizer(self, regularizer) -> None:
        """
        Adds a regularizer to the optimizer.

        Args:
            regularizer: The regularizer to add.
        """
        self.regularizer = regularizer

    def calculate_update(
        self, weight_tensor: np.ndarray, gradient_tensor: np.ndarray
    ) -> np.ndarray:
        """
        Calculate an update to the weight tensor.

        Args:
            weight_tensor: The weight tensor to update.
            gradient_tensor: The gradient tensor to use for the update.

        Returns:
            The updated weight tensor.
        """
        if self.regularizer is not None:
            gradient_tensor += self.regularizer.calculate_gradient(weight_tensor)
        return weight_tensor


# Basic optimizer that performs a single step of gradient descent
# on the given weight tensor.
class Sgd(Optimizer):
    """
    Performs a single step of gradient descent on the given weight tensor.

    Args:
        learning_rate: The learning rate to use.

    Returns:
        The updated weight tensor.
    """

    def __init__(self, learning_rate: float) -> None:
        super().__init__(learning_rate)

    def calculate_update(
        self, weight_tensor: np.ndarray, gradient_tensor: np.ndarray
    ) -> np.ndarray:
        return weight_tensor - self.learning_rate * gradient_tensor


class SgdWithMomentum(Optimizer):
    """
    Performs a single step of gradient descent with momentum on the given weight tensor.

    Args:
        learning_rate: The learning rate to use.
        momentum: The momentum factor.
        weight_tensor: The weight tensor to update.
        gradient_tensor: The gradient tensor to use for the update.
        velocity: The velocity tensor to use for the update.

    Returns:
        The updated weight tensor.
    """

    def __init__(self, learning_rate: float, momentum: float) -> None:
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = 0

    def calculate_update(
        self, weight_tensor: np.ndarray, gradient_tensor: np.ndarray
    ) -> np.ndarray:
        '''
        It calculates the updated weight tensor using the formula:
        weight_tensor + velocity
        where velocity is calculated as:
        momentum * velocity - learning_rate * gradient_tensor
        
        Args:
            weight_tensor: The weight tensor to update.
            gradient_tensor: The gradient tensor to use for the update. 
            
        Returns:
            The updated weight tensor.
        '''
        self.velocity = (
            self.momentum * self.velocity - self.learning_rate * gradient_tensor
        )
        return weight_tensor + self.velocity


class Adam(Optimizer):
    """
    Performs a single step of Adam on the given weight tensor.

    Args:
        learning_rate: The learning rate to use.
        mu: The momentum factor.
        rho: The decay factor.
        r: The running average of the squared gradient.
        v: The running average of the gradient.
        k: The current iteration.      
    
    Returns:
        The updated weight tensor.
    """

    def __init__(self, learning_rate: float, mu: float, rho: float) -> None:
        super().__init__(learning_rate)
        self.mu = mu
        self.rho = rho
        self.epsilon = 1e-8
        self.r = 0
        self.v = 0
        self.k = 1

    def calculate_update(
        self, weight_tensor: np.ndarray, gradient_tensor: np.ndarray
    ) -> np.ndarray:
        """
        Perform a single step of Adam on the given weight tensor.
        Args:
            weight_tensor: The weight tensor to update.
            gradient_tensor: The gradient tensor to use for the update.
        Returns:
            The updated weight tensor.
        """
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor # first moment estimate
        self.r = self.rho * self.r + (1 - self.rho) * (gradient_tensor**2) # second raw moment estimate

        # Bias correction for the first and second moment estimates
        v_hat = self.v / (1 - self.mu**self.k)
        r_hat = self.r / (1 - self.rho**self.k)
        self.k += 1
        return weight_tensor - self.learning_rate * v_hat / (
            np.sqrt(r_hat) + self.epsilon
        )
