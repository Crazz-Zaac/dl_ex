import numpy as np
from copy import deepcopy

class NeuralNetwork:
    """
    defines the whole neural network architecture containing all its layers 
    from input to output layer
    """
    
    def __init__(self, optimizer: np.ndarray) -> None:
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
    
    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        '''
        forward pass through the network
        Args:
            input_tensor: np.ndarray
                input tensor for the forward pass
        Returns:
            np.ndarray 
                output tensor after applying the forward pass
        '''
        input_tensor, self.label_tensor = self.data_layer.next()
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        self.prediction = self.loss_layer.forward(input_tensor, label_tensor)
        return self.prediction
    
    def backward(self) -> None:
        '''
        backward pass through the network
        '''
        error_tensor = self.loss_layer.backward(self.label_tensor)
        
        # iterating through all the layers in reversed order
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor) 
    
    
    def append_layer(self, layer: list) -> None:
        '''
        append the layer to the list of layers in the network
        Args:
            layer: object
                layer to be appended
        '''
        # if the layer is trainable then it makes a deep copy of the optimizer 
        # and assigns it to the layer
        if layer.trainable:
            layer.optimizer = deepcopy(self.optimizer)
                
        self.layers.append(layer)
        
    
    def train(self, iterations: int) -> None:
        '''
        train the network for the given number of iterations
        Args:
            iterations: int
                number of iterations to train the network
        '''
        for _ in range(iterations):
            self.forward(self.data_layer)
            self.backward()
            self.loss.append(self.loss_layer.calculate_loss(self.prediction, self.label_tensor))
            print(f'Loss: {self.loss[-1]}')
    
    def test(self, input_tensor: np.ndarray) -> np.ndarray:
        '''
        test the network on the input tensor
        Args:
            input_tensor: np.ndarray
                input tensor for the network
        Returns:
            np.ndarray
                output tensor after testing the network
        '''
        for layer in self.layers:
            output_predict = self.forward(input_tensor)
        return output_predict