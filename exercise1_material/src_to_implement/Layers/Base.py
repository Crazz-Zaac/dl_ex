
# Base class for all layers.
class BaseLayer:
    
    def __init__(self) -> None:
        self.trainable = False
        self.input_tensor = None
        self.output_tensor = None