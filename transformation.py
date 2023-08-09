import numpy as np 
from enum import Enum 

class TransformType(Enum): 
        ROTATION = 0 
        XREFLECTION = 1 
        YREFLECTION = 2
        ZREFLECTION = 3
class Transformation: 
    def __init__(self, type: TransformType, matrix : np.array = None, confidence : float = 0, angle : float = 0) -> None:
        self.type = type 
        self.matrix = matrix
        self.confidence = confidence
        self.angle = angle

    def setConfidence(self, confidence : float): 
        self.confidence = confidence
    
    def getConfidence(self):  
        return self.confidence
    