import numpy as np
from src_to_implement.Optimization.Optimizer import Optimizer


class Sgd:
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate

    def update(self, w: np.ndarray, grad_wrt_w: np.ndarray) -> np.ndarray:
        return w - self.learning_rate * grad_wrt_w

    def get_name(self):
        return "sgd"