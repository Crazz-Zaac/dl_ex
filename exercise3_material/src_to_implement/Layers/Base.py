
# Base class for all layers.
class BaseLayer:
    
    def __init__(self) -> None:
        self.trainable = False
        self.weights = None
        self.testing_phase = False