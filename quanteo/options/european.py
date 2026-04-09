import numpy as np
from options.base_option import BaseOption

class EuropeanOption(BaseOption):
    """
    European Option. This class is child of BaseOption, i.e. the contract information (T, K, put/call) are taken from it.
    Reminder: European Option can be exercised only at maturity (T). 
    """

    #constructor:
    def __init__(self, T: float, K: float, option_type: str = "call"):
        super().__init__(T, K, option_type)


    # Payoff function for european options
    def payoff(self, paths: np.ndarray) -> np.ndarray:
        # Extract S_T (the forecasted asset price at maturity) for each trajectory simulated. Manage the cases where S_T is a single point
        S_T = paths[:, -1] if paths.ndim > 1 else paths
        
        # From the array of S_T's -> array of payoffs. Later on will be extracted average payoff + CI
        if self.option_type == "call":
            return np.maximum(0, S_T - self.K)
        elif self.option_type == "put":
            return np.maximum(0, self.K - S_T)

