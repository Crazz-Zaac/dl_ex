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
        mu: The mu factor.
        rho: The rho factor.
        epsilon: The epsilon factor.
        weight_tensor: The weight tensor to update.
        gradient_tensor: The gradient tensor to use for the update.
        m: The first moment tensor to use for the update.
        v: The second moment tensor to use for the update.
        mu_t: The mu factor to the power of t.
        rho_t: The rho factor to the power of t.

    Returns:
        The updated weight tensor.
    """

    def __init__(self, learning_rate: float, mu: float, rho: float) -> None:
        self.learning_rate = learning_rate
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
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1 - self.rho) * (gradient_tensor**2)

        v_hat = self.v / (1 - self.mu**self.k)
        r_hat = self.r / (1 - self.rho**self.k)
        self.k += 1
        return weight_tensor - self.learning_rate * v_hat / (
            np.sqrt(r_hat) + self.epsilon
        )
