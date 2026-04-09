import numpy as np
from abc import ABC, abstractmethod

class BaseOption(ABC):
    """
    Abstract base class for option contracts.
    It stores contract specifications and mandates a payoff implementation.
    """
    def __init__(self, T: float, K: float, option_type: str = "call"):
        """
        Constructor
        Args:
            
            T (float): Option maturity (e.g. 1.0 for 1 year).
            K (float): Strike price.
            option_type (str, optional): call or put. Defaults to "call".
        """
        self.T = T
        self.K = K
        self.option_type = option_type.lower()
        
        # AVOID SYNTAX ERRORS 
        if self.option_type not in ["call", "put"]:
            raise ValueError("Option name is not valid: only 'put' and 'call' are allowed")
        if self.T < 0:
            raise ValueError("T must be >= 0")
        if self.K <= 0:
            raise ValueError("K must be > 0")
    

    @abstractmethod
    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """
        Calculates the option payoff given the simulated asset paths.
        Must be implemented by child classes.
        """
        pass