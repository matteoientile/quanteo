import numpy
from abc import ABC, abstractmethod

class BaseRisk(ABC):
    """
    Abstract class for Greeks calculation of any kind of option
    """
    @abstractmethod
    def greeks_calculator(self, option, model) -> dict:
        """
        Abstract method to compute Delta, Gamma, Vega, Theta and Rho. 
        Returns a dictionary.
        """
        pass
