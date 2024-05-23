import numpy as np 

class CrossEntropyLoss:
    """
    Used to calculate the cross entropy loss between the predicted and true labels.
    Typically in conjunction with Softmax or sigmoid activation function
    """
        
    def __init__(self):
        self.output_tensor = None
    
    def forward(self, prediction_tensor: np.ndarray, label_tensor: np.ndarray) -> np.ndarray:
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
        self.output_tensor = -np.sum(label_tensor * np.log(prediction_tensor))
        return self.output_tensor
    
    def backward(self, label_tensor: np.ndarray) -> np.ndarray:
        """
        Computes the backward pass for cross entropy loss.
        
        Returns:
            np.ndarray
                The error tensor after applying the backward pass.
        """
        output_tensor = -label_tensor / self.prediction_tensor
        return self.output_tensor