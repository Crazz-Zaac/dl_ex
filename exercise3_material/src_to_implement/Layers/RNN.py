import numpy as np
from .Base import BaseLayer
from .FullyConnected import FullyConnected
from .TanH import TanH
from .Sigmoid import Sigmoid


class RNN(BaseLayer):

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.trainable = True
        
        self.regularization_loss = 0

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initializing the hidden state with zeros
        self.hidden_state = np.zeros((hidden_size))

        self._memorize = False
        self._optimizer = None
        self._gradient_weights = None

        self.tanh = TanH()
        self.sigmoid = Sigmoid()

        # Fully connected layer for the hidden state and the output layer
        self.hidden_fcl = FullyConnected(
            self.input_size + self.hidden_size, self.hidden_size
        )
        self.output_fcl = FullyConnected(self.hidden_size, self.output_size)

        # For memorizing the intermediate values during the forward pass
        self.hidden_fcl_input_tensor = []
        self.output_fcl_input_tensor = []
        self.sigmoid_outputs = []
        self.tanh_outputs = []

        self.hidden_fcl_gradient_weights = []
        self.output_fcl_gradient_weights = []
        
        

        

    def calculate_regularization_loss(self):
        if self.optimizer.regularizer:
            self.regularization_loss += self.optimizer.regularizer.norm(
                self.hidden_fcl.weights
            ) + self.optimizer.regularizer.norm(self.output_fcl.weights)

        return self.regularization_loss

    def initialize(self, weights_initializer, bias_initializer):
        self.hidden_fcl.initialize(weights_initializer, bias_initializer)
        self.output_fcl.initialize(weights_initializer, bias_initializer)

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        self.sigmoid_outputs.clear()
        self.tanh_outputs.clear()
        self.hidden_fcl_input_tensor.clear()
        self.output_fcl_input_tensor.clear()

        previous_hidden_state = (
            self.hidden_state if self.memorize else np.zeros(self.hidden_size)
        )
        batch_size = input_tensor.shape[0]
        output_tensor = np.zeros((batch_size, self.output_size))

        for i in range(batch_size):
            combined_input = np.concatenate(
                (previous_hidden_state,input_tensor[i])
            ).reshape(1, -1)
            hidden_fcl_input = self.hidden_fcl.forward(combined_input)
            tanh_output = self.tanh.forward(hidden_fcl_input)

            previous_hidden_state = tanh_output[0]  # tanh_output is a 2D array

            output_in = self.output_fcl.forward(tanh_output)
            sigmoid_out = self.sigmoid.forward(output_in)
            output_tensor[i] = sigmoid_out[0]

            self.hidden_fcl_input_tensor.append(self.hidden_fcl.input)
            self.output_fcl_input_tensor.append(self.output_fcl.input)
            self.sigmoid_outputs.append(self.sigmoid.activation)
            self.tanh_outputs.append(self.tanh.activation)

        self.hidden_state = previous_hidden_state
        return output_tensor

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        self.gradient_weights = np.zeros_like(self.hidden_fcl.weights)
        self.output_fcl_gradient_weights = np.zeros_like(self.output_fcl.weights)

        gradient_previous_hidden_state = 0 # np.zeros(self.hidden_size)
        batch_size = error_tensor.shape[0]
        gradient_wrt_inputs = np.zeros((batch_size, self.input_size))

        for step in range(batch_size-1, -1, -1):
            self.sigmoid.activation = self.sigmoid_outputs[step]
            sigmoid_error = self.sigmoid.backward(error_tensor[step])

            self.output_fcl.input = self.output_fcl_input_tensor[step]
            output_fcl_error = self.output_fcl.backward(sigmoid_error)

            self.tanh.activation = self.tanh_outputs[step]
            tanh_error = self.tanh.backward(
                output_fcl_error + gradient_previous_hidden_state
            )

            self.hidden_fcl.input = self.hidden_fcl_input_tensor[step]
            hidden_fcl_error = self.hidden_fcl.backward(tanh_error)

            gradient_previous_hidden_state = hidden_fcl_error[:, : self.hidden_size]
            gradient_wrt_inputs[step] = hidden_fcl_error[:, self.hidden_size :]

            self.gradient_weights += self.hidden_fcl.gradient_weights
            self.output_fcl_gradient_weights += self.output_fcl.gradient_weights

        if self.optimizer:
            # self.hidden_fcl.weights = self.optimizer.calculate_update(
            #     self.hidden_fcl.weights, self.hidden_fcl.gradient_weights
            # )
            self.output_fcl.weights = self.optimizer.calculate_update(
                self.output_fcl.weights, self.output_fcl_gradient_weights
            )
            self.weights = self.optimizer.calculate_update(
                self.weights, self.gradient_weights
            )

        return gradient_wrt_inputs
    
    

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        # self.hidden_fcl.optimizer = optimizer
        # self.output_fcl.optimizer = optimizer

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, memorize):
        self._memorize = memorize

    @property
    def weights(self):
        return self.hidden_fcl.weights

    @weights.setter
    def weights(self, weights):
        self.hidden_fcl.weights = weights

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights