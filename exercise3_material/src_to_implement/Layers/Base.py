
# Base class for all layers.
class BaseLayer:
    '''
    Base class for all layers.
    '''
    
    def __init__(self) -> None:
        # Initialize the trainable parameter to False
        # Initialize the testing_phase parameter to False
        self.trainable = False
        self.testing_phase = False