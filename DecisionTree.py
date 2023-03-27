from dataclasses import dataclass
import numpy as np 

@dataclass
class description:
    name:str
    maxValue: float
    minValue: float
    Q1: float
    Q2: float
    Q3: float
    Q4: float
    mean: float



class DecisionTree:
    def __init__(self) -> None:
        pass
    
    def fit() -> None:
        pass
    
    def predict() -> np.ndarray:
        pass
