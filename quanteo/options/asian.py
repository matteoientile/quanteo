import numpy as np 
from options import BaseOption

class ArithmeticAsianOption(BaseOption):
    """
    Compute the payoff for Arithmetic Asian Options
    """
    def __init__(self, T: float, K: float, N: int, option_type: str = "call"):
        """
        Args:
            T (float): Option maturity (e.g. 1.0 for 1 year).
            K (float): Strike price.
            N (int): Number of prices to take into account for arithmetic average
            option_type (str, optional): call or put. Defaults to "call".
        """
        super().__init__(T=T, K=K, option_type=option_type) 
        self.N = N #how many prices to take into account. S0 is not considered for the average


    #======================================
    #========= PUBLIC METHODS =============
    #======================================
    def payoff(self, paths: np.ndarray) -> np.ndarray:
        array_S = paths[:, 1:self.N+1] 
        arithmetic_avgs = 1/(self.N) * np.sum(array_S, axis=1)

        if self.option_type == "call":
            return np.maximum(0, arithmetic_avgs - self.K)
        elif self.option_type == "put":
            return np.maximum(0, self.K - arithmetic_avgs)

    

class GeometricAsianOption(BaseOption):
    """
    Compute the payoff for Geometric Asian Options
    """
    def __init__(self, T: float, K: float, N: int, option_type: str = "call"):
        """
        Args:
            T (float): Option maturity (e.g. 1.0 for 1 year).
            K (float): Strike price.
            N (int): Number of prices to take into account for arithmetic average
            option_type (str, optional): call or put. Defaults to "call".
        """
        super().__init__(T=T, K=K, option_type=option_type) 
        self.N = N #how many prices to take into account. S0 is not considered for the average


    #======================================
    #========= PUBLIC METHODS =============
    #======================================
    def payoff(self, paths: np.ndarray) -> np.ndarray:
        array_S = paths[:, 1:self.N+1] 
        log_S = np.log(array_S)

        geometric_avgs = np.exp(np.mean(log_S, axis=1))

        if self.option_type == "call":
            return np.maximum(0, geometric_avgs - self.K)
        elif self.option_type == "put":
            return np.maximum(0, self.K - geometric_avgs)
