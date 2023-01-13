import numpy as np

    
class Interpreter(object):
    
    def __init__(self, model) -> None:
        self.model = model
    
    def __call__(self, image_path:str) -> np.ndarray:
        pass

