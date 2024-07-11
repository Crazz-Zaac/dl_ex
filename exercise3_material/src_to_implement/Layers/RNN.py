import numpy as np
from .Base import BaseLayer
from .FullyConnected import FullyConnected
from .TanH import TanH
from .Sigmoid import Sigmoid


class RNN(BaseLayer):
    '''
    RNN layer implementation using the following formula:
    h_t = tanh(W_h * [h_t-1, x_t] + b_h)
    y_t = sigmoid(W_y * h_t + b_y)
    where:
    h_t is the hidden state at time t
    x_t is the input at time t
    y_t is the output at time t
    W_h, W_y are the weights of the hidden and output layer respectively
    b_h, b_y are the biases of the hidden and output layer respectively
    
    Attributes:
    input_size: int - size of the input tensor
    hidden_size: int - size of the hidden state
    output_size: int - size of the output tensor
    hidden_state: np.ndarray - hidden state of the RNN layer
    memorize: bool - flag to memorize the hidden state across forward passes
    optimizer: Optimizer - optimizer that defines the optimization algorithm to be used for updating the weights
    regularization_loss: float - regularization loss
    gradient_weights: np.ndarray - gradient weights of the hidden and output fully connected layers
    tanh: TanH - tanh activation function
    sigmoid: Sigmoid - sigmoid activation function
    hidden_fcl: FullyConnected - fully connected layer for the hidden state
    output_fcl: FullyConnected - fully connected layer for the output tensor
    hidden_fcl_input_tensor: list - list to store the input tensor of the hidden fully connected layer
    output_fcl_input_tensor: list - list to store the input tensor of the output fully connected layer
    sigmoid_outputs: list - list to store the output of the sigmoid activation function
    tanh_outputs: list - list to store the output of the tanh activation function
    hidden_fcl_gradient_weights: list - list to store the gradient weights of the hidden fully connected layer
    output_fcl_gradient_weights: list - list to store the gradient weights of the output fully connected layer
    
    Methods:
    calculate_regularization_loss: Calculate the regularization loss for the hidden and output layer weights
    initialize: Initialize the weights and biases of the hidden and output layer using the provided initializer
    forward: Calculate the forward pass of the RNN layer
    backward: Calculate the backward pass of the RNN layer    
    '''
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.trainable = True
        
        self.regularization_loss = 0

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initializing the hidden state with zeros
        self.hidden_state = np.zeros((hidden_size))

        # a flag to decide whether to memorize the hidden state across forward passes
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
       

    def calculate_regularization_loss(self) -> float:
        '''
        Calculate the regularization loss for the hidden and output layer weights 
        using the optimizer's regularizer
        If regularizer is provided, the regularization loss is calculated as:
            - regularization_loss = regularization_loss + norm(W_h) + norm(W_y)
        else
            - regularization_loss = 0        
        Returns:
        regularization_loss: float - regularization loss
        
        '''        
        if self.optimizer.regularizer:
            self.regularization_loss += self.optimizer.regularizer.norm(
                self.hidden_fcl.weights
            ) + self.optimizer.regularizer.norm(self.output_fcl.weights)

        return self.regularization_loss

    
    def initialize(self, weights_initializer, bias_initializer) -> None:
        '''
        Initialize the weights and biases of the hidden and output layer using the 
        provided initializer
        '''
        self.hidden_fcl.initialize(weights_initializer, bias_initializer)
        self.output_fcl.initialize(weights_initializer, bias_initializer)

    
    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        '''
        This function calculates the forward pass of the RNN layer.
        - Clear the intermediate values stored during the forward pass
         
        - Initialize the hidden state with zeros if memorize is set to False
        - Get the batch_size from the input tensor
        - Initialize the output tensor with zeros
        
        For each batch_size:
        - combined_input = [previous_hidden_state, input_tensor]
        - hidden_fcl_input = W_h * combined_input + b_h
        - tanh_output = tanh(hidden_fcl_input)
        - output_in = W_y * tanh_output + b_y
        - sigmoid_out = sigmoid(output_in)
        
        Arguments:
        input_tensor: np.ndarray - input tensor of shape (batch_size, input_size)
        
        Returns:
        output_tensor: np.ndarray - output tensor of shape (batch_size, output_size)
        '''
        
        # Clear the intermediate values stored during the forward pass
        self.sigmoid_outputs.clear()
        self.tanh_outputs.clear()
        self.hidden_fcl_input_tensor.clear()
        self.output_fcl_input_tensor.clear()

        # Initialize the hidden state with zeros if memorize is set to False
        previous_hidden_state = (
            self.hidden_state if self.memorize else np.zeros(self.hidden_size)
        )
        
        # Get the batch_size from the input tensor and initialize the output tensor with zeros
        batch_size = input_tensor.shape[0]
        output_tensor = np.zeros((batch_size, self.output_size))

        # For each batch_size calculate the forward pass
        for i in range(batch_size):
            combined_input = np.concatenate(
                (previous_hidden_state, input_tensor[i])
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
        '''
        This function calculates the backward pass of the RNN layer.
        - initialize the gradient weights of the hidden and output fully connected layers with zeros
        - set the gradient_previous_hidden_state to zero
        - get the batch_size from the error tensor
        - initialize the gradient_wrt_inputs with zeros
                
        For each batch_size (in reverse order):
        - Calculate the gradients of the loss L w.r.t the error tensor (y_t - y_hat_t)
        - Propagate back through the output fully connected layer and compute gradient w.r.t the hidden state (h_t)
        - Compute gradient of the loss w.r.t the hidden state (h_t) and the deravative of the tanh activation from next time step (h_t+1)
        - Propagate back through the hidden fully connected layer and compute gradient w.r.t combined_input
        - Accumulate the gradients of the hidden and output fully connected layers
        - If an optimizer is provided, update the weights of the hidden and output fully connected layers
                
        Arguments:
        error_tensor: np.ndarray - error tensor of shape (batch_size, output_size)
        
        Returns:
        gradient_wrt_inputs: np.ndarray - gradient of the loss w.r.t the input tensor of shape (batch_size, input_size)
         
        
        '''
        self.gradient_weights = np.zeros_like(self.hidden_fcl.weights)
        self.output_fcl_gradient_weights = np.zeros_like(self.output_fcl.weights)

        gradient_previous_hidden_state = 0 # np.zeros(self.hidden_size)
        
        # Get the batch_size from the error tensor and initialize the gradient_wrt_inputs with zeros
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
    
    
    # setter and getter methods for the optimizer that defines the optimization algorithm
    # to be used for updating the weights of the hidden and output fully connected layers
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    
    # setter and getter methods for the memorize attribute that defines whether the hidden state
    # should be memorized across forward passes
    # if memorize is set to True, the hidden state from the previous forward pass is used as the initial hidden state
    # else the hidden state is initialized with zeros
    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, memorize):
        self._memorize = memorize

    
    # setter and getter methods for the weights of the hidden and output fully connected layers
    @property
    def weights(self):
        return self.hidden_fcl.weights

    @weights.setter
    def weights(self, weights):
        self.hidden_fcl.weights = weights
        
    
    # setter and getter methods for the gradient weights of the hidden and output fully connected layers
    # that are used to update the weights of the hidden and output fully connected layers
    # during the backward pass
    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights