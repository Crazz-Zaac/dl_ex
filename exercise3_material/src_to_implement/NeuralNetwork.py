import numpy as np
from copy import deepcopy


class NeuralNetwork:
    """
    It defines the neural network class which is used to train and test the network.

    Attributes:
        optimizer: np.ndarray
            optimizer to be used for the network
        loss: list
            list to store the loss after each iteration
        layers: list
            list to store the layers of the network
        data_layer: object
            data layer object
        loss_layer: object
            loss layer object
    """

    def __init__(
        self, optimizer: np.ndarray, weights_initializer, bias_initializer
    ) -> None:
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self._phase = None

    def forward(self) -> np.ndarray:
        """
        It calculates the forward pass through the network
        and calculates the loss using the loss layer.
        For each layer in the network:
            - calculate the forward pass
            - if the layer is trainable, calculate the regularization loss
            - set the testing_phase to True
            - calculate the loss using the loss layer

        Args:
            input_tensor: np.ndarray
                input tensor for the forward pass
                Shape: (batch_size, input_size)
        Returns:
            np.ndarray
                output tensor after applying the forward pass
                shape: (batch_size, output_size)
        """
        input_tensor, self.label_tensor = self.data_layer.next()
        reg_loss = 0
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
            try:
                reg_loss += self.optimizer.regularizer.norm(layer.weights)
            except:
                pass
            layer.testing_phase = True
        self.prediction = self.loss_layer.forward(
            input_tensor + reg_loss, self.label_tensor
        )
        return self.prediction

    def backward(self) -> None:
        """
        It calculates the backward pass through the network
        and updates the weights of the layers.
        For layer in reversed order:
            - calculate the backward pass
            - update the weights of the layer
            - if the layer is trainable, update the weights using the optimizer

        """
        loss_ = self.loss_layer.backward(self.label_tensor)

        # iterating through all the layers in reversed order
        for layer in self.layers[::-1]:
            loss_ = layer.backward(loss_)

    def append_layer(self, layer) -> None:
        """
        append the layer to the list of layers in the network
        Args:
            layer: object
                layer to be appended
                shape: (batch_size, output_size)
        """
        # if the layer is trainable then it makes a deep copy of the optimizer
        # and assigns it to the layer
        if layer.trainable:
            layer.optimizer = deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)

        self.layers.append(layer)

    def train(self, iterations: int) -> None:
        """
        train the network for the given number of iterations
        Args:
            iterations: int
                number of iterations to train the network
        """
        for _ in range(iterations):
            loss_ = self.forward()
            self.backward()
            self.loss.append(loss_)

    def test(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        This function is used to test the network using the input tensor
        Args:
            input_tensor: np.ndarray
                input tensor for the network
        Returns:
            np.ndarray
                output tensor after testing the network
        """
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor

    # getter and setter for the phase attribute
    # the phase attribute is used to set the phase of the network
    # whether it is training or testing
    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase):
        self._phase = phase
