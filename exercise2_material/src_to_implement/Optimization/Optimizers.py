import numpy as np


# Basic optimizer that performs a single step of gradient descent
# on the given weight tensor.
class Sgd:
    """
    Performs a single step of gradient descent on the given weight tensor.

    Args:
        learning_rate: The learning rate to use.

    Returns:
        The updated weight tensor.
    """

    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate

    def calculate_update(
        self, weight_tensor: np.ndarray, gradient_tensor: np.ndarray
    ) -> np.ndarray:
        return weight_tensor - self.learning_rate * gradient_tensor


class SgdWithMomentum:
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
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = 0

    def calculate_update(
        self, weight_tensor: np.ndarray, gradient_tensor: np.ndarray
    ) -> np.ndarray:
        self.velocity = (
            self.momentum * self.velocity - self.learning_rate * gradient_tensor
        )
        return weight_tensor + self.velocity


class Adam:
    """
    Performs a single step of Adam on the given weight tensor.

    Args:
        learning_rate: The learning rate to use.
        beta_1: The beta_1 factor.
        beta_2: The beta_2 factor.
        epsilon: The epsilon factor.
        weight_tensor: The weight tensor to update.
        gradient_tensor: The gradient tensor to use for the update.
        m: The first moment tensor to use for the update.
        v: The second moment tensor to use for the update.
        beta_1_t: The beta_1 factor to the power of t.
        beta_2_t: The beta_2 factor to the power of t.

    Returns:
        The updated weight tensor.
    """

    def __init__(
        self, learning_rate: float, beta_1: float, beta_2: float, epsilon: float
    ) -> None:
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.beta_1_t = beta_1
        self.beta_2_t = beta_2

    def calculate_update(
        self, weight_tensor: np.ndarray, gradient_tensor: np.ndarray
    ) -> np.ndarray:
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient_tensor
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * gradient_tensor**2
        self.beta_1_t *= self.beta_1
        self.beta_2_t *= self.beta_2
        m_hat = self.m / (1 - self.beta_1_t)
        v_hat = self.v / (1 - self.beta_2_t)
        return weight_tensor - self.learning_rate * m_hat / (
            np.sqrt(v_hat) + self.epsilon
        )
