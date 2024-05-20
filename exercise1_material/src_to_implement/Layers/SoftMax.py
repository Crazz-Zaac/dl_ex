import numpy as np  

class SoftMax:
    
    """
    Transforms the input tensor into a tensor of probabilities using the softmax function.
    """
    
    def __init__(self):
        self.output_tensor = None
    
    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass for a softmax function.
        
        Args:
            input_tensor: np.ndarray
                The input tensor for the forward pass.
        
        Returns:
            np.ndarray
                The output tensor after applying the softmax function.
        """
        # shifting the input tensor to avoid numerical instability
        exp_input_tensor = np.exp(input_tensor - np.max(input_tensor))
        self.output_tensor = exp_input_tensor / np.sum(exp_input_tensor, axis=1, keepdims=True)
        return self.output_tensor
    
    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        """
        Computes the backward pass for a softmax function.
        
        Args:
            error_tensor: np.ndarray
                The error tensor for the backward pass.
        
        Returns:
            np.ndarray
                The error tensor after applying the backward pass.
        """
        batch_size, num_classes = error_tensor.shape
        gradient_input = np.zeros_like(error_tensor)
        
        for i in range(batch_size):
            # softmax output for the current sample
            y = self.output_tensor[i]
            
            # error tensor for the current sample
            e = error_tensor[i]
            
            gradient_input = y * e - y * np.sum(y * e)

        return gradient_input
    
