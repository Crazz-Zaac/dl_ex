import numpy as np

class L2_Regularizer:
    """
    L2 Regularization class to calculate the gradient of the L2 norm.
    """

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def calculate_gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        Calculate the gradient of the L2 norm.
        Args:
            weights: The weight tensor to calculate the gradient for.
        Returns:
            The gradient of the L2 norm.
        """
        return 2 * self.alpha * weights
    
    def norm(self, weights: np.ndarray) -> np.ndarray:
        """
        Calculate the L2 norm.
        Args:
            weights: The weight tensor to calculate the norm for.
        Returns:
            The L2 norm.
        """
        return self.alpha * (weights ** 2).sum()

class L1_Regularizer:
    """
    L1 Regularization class to calculate the gradient of the L1 norm.
    """

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def calculate_gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        Calculate the gradient of the L1 norm.
        Args:
            weights: The weight tensor to calculate the gradient for.
        Returns:
            The gradient of the L1 norm.
        """
        return self.alpha * (weights > 0).astype(float) - self.alpha * (weights < 0).astype(float)
    
    def norm(self, weights: np.ndarray) -> np.ndarray:
        """
        Calculate the L1 norm.
        Args:
            weights: The weight tensor to calculate the norm for.
        Returns:
            The L1 norm.
        """
        return self.alpha * np.abs(weights).sum()