import numpy as np

class L2_Regularizer:
    """
    L2 Regularization class to calculate the gradient of the L2 norm.
    """

    def __init__(self, regularization_rate: float) -> None:
        self.regularization_rate = regularization_rate

    def calculate_gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        Calculate the gradient of the L2 norm.
        Args:
            weights: The weight tensor to calculate the gradient for.
        Returns:
            The gradient of the L2 norm.
        """
        return 2 * self.regularization_rate * weights
    
    def norm(self, weights: np.ndarray) -> np.ndarray:
        """
        Calculate the L2 norm.
        Args:
            weights: The weight tensor to calculate the norm for.
        Returns:
            The L2 norm.
        """
        return self.regularization_rate * np.linalg.norm(weights)

class L1_Regularizer:
    """
    L1 Regularization class to calculate the gradient of the L1 norm.
    """

    def __init__(self, regularization_rate: float) -> None:
        self.regularization_rate = regularization_rate

    def calculate_gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        Calculate the gradient of the L1 norm.
        Args:
            weights: The weight tensor to calculate the gradient for.
        Returns:
            The gradient of the L1 norm.
        """
        return self.regularization_rate * np.sign(weights)
    
    def norm(self, weights: np.ndarray) -> np.ndarray:
        """
        Calculate the L1 norm.
        Args:
            weights: The weight tensor to calculate the norm for.
        Returns:
            The L1 norm.
        """
        return self.regularization_rate * np.sum(np.abs(weights))