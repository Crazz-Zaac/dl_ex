import numpy as np


class CrossEntropyLoss:
    """
    Used to calculate the cross entropy loss between the predicted and true labels.
    Typically in conjunction with Softmax or sigmoid activation function
    
    Attributes:
        epsilon: np.float64
            A small value to avoid division by zero
        prediction_tensor: np.ndarray
            The input tensor for the forward pass
        entropy_loss: np.float64
            The output tensor after applying the cross entropy loss
        
    """
    

    def __init__(self):
        self.epsilon = np.finfo(float).eps

    def forward(
        self, prediction_tensor: np.ndarray, label_tensor: np.ndarray
    ) -> np.ndarray:
        """
        Computes the forward pass for cross entropy loss.

        Args:
            prediction_tensor: np.ndarray
                The input tensor for the forward pass.
            label_tensor: np.ndarray
                The label tensor for the forward pass.

        Returns:
            np.ndarray
                The output tensor after applying the cross entropy loss.
        """
        
        # to avoid division by zero when taking the log
        self.prediction_tensor = prediction_tensor 
        
        loss_ = self.prediction_tensor + self.epsilon


        # get the indices of the maximum values within each row of the label tensor        
        true_class_indices = np.argmax(label_tensor, axis=1)
        # gets the probabilities of the classes from the loss_ array using the true class indices
        true_class_probs = loss_[
            np.arange(label_tensor.shape[0]), true_class_indices
        ]
        
        # calculate the cross entropy loss
        self.entropy_loss = -np.sum(np.log(true_class_probs))
        
        return self.entropy_loss

    def backward(self, label_tensor: np.ndarray) -> np.ndarray:
        """
        Computes the backward pass for cross entropy loss.
        
        Args:
            label_tensor: np.ndarray
                The label tensor for the backward pass.

        Returns:
            np.ndarray
                The error tensor after applying the backward pass.
        """
        # for numerical stability, add a small value to the prediction tensor
        loss_ = self.prediction_tensor + self.epsilon
        
        error_tensor = -label_tensor / loss_
        
        return error_tensor
