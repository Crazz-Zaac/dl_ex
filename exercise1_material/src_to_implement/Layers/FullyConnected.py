import numpy as np
from Layers.Base import BaseLayer



class FullyConnected(BaseLayer):
    
    def __init__(self, input_size: int, output_size: int) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(0, 1, (input_size, output_size))
        self.bias = np.zeros(output_size)
        self.trainable = True
        
    
    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        batch_size = input_tensor.shape[0]
        self.input_tensor = np.random.randn(batch_size, self.input_size)
        self.output_tensor = np.dot(input_tensor, self.weights)
        return self.output_tensor
    
    # setter and getter property optimizer
    @property
    def optimizer(self):
        return self.optimizer
        
    @optimizer.getter
    def optimizer(self):
        return self.optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self.optimizer = optimizer
        self.optimizer.register_layer(self)
    
    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        cache = np.dot(error_tensor, self.weights.T)
        self.optimizer.update(self)
        self.weights = self.optimizer.update(self.weights)
        self.bias = self.optimizer.update(self.bias)
        return cache
    
    