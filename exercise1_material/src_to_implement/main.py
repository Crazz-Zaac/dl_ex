import numpy as np
# from NeuralNetworkTests import TestFullyConnected1, TestReLU, TestSoftMax
from Layers import ReLU, SoftMax
from Layers.FullyConnected import FullyConnected
from NeuralNetwork import NeuralNetwork
from Optimization.Optimizers import Sgd
from Loss import CrossEntropyLoss


class DataGenerator:
    
    def __init__(self, rows, cols, batch_size, output_size):
        self.rows = rows
        self.cols = cols
        self.batch_size = batch_size
        self.X = np.random.randn(rows, cols)
        self.y = np.random.randn(rows, output_size)
        
    def next(self):
        idx = np.random.choice(self.rows, self.batch_size)
        return self.X[idx], self.y[idx]

# Define the input size and output size
input_size = 5
output_size = 3
batch_size = 4



if __name__ == "__main__":
    
    

    # # Forward pass through the layer
    # output_tensor = fc_layer.forward(input_tensor)

    # # Compute loss (just for testing purposes, you can replace it with your actual loss computation)
    # loss = np.mean((output_tensor - label_tensor) ** 2)

    # # Backward pass through the layer
    # error_tensor = 2 * (output_tensor - label_tensor) / batch_size  # Gradient of mean squared error loss
    # gradient_tensor = fc_layer.backward(error_tensor)

    # # Update weights using the optimizer
    # fc_layer.optimizer = optimizer  # Set optimizer for the layer
    # updated_weights = fc_layer.weights  # Updated weights after optimization

    optimizer = Sgd(learning_rate=0.01)

    nn = NeuralNetwork(optimizer)
    nn.data_layer = DataGenerator(100, 3, 4, 2)
    nn.loss_layer = CrossEntropyLoss()
    
    # Create an instance of FullyConnected layer and Sgd optimizer
    fc_layer = FullyConnected(3, 4)
    nn.append_layer(fc_layer)
    nn.append_layer(FullyConnected(4, 2))   
    nn.train(10)
    
    
   

    # Print the shapes of generated tensors for verification
    print("Input tensor shape:", input_tensor.shape)
    print("Label tensor shape:", label_tensor.shape)
    print("Output tensor shape:", output_tensor.shape)
    print("Gradient tensor shape:", gradient_tensor.shape)
   
