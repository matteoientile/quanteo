import numpy as np
from quanteo.options.base_option import BaseOption

class EuropeanOption(BaseOption):
    """
    Contract definition for a standard European Option.

    A European option is a vanilla financial derivative that grants the holder 
    the right, but not the obligation, to buy (call) or sell (put) an underlying 
    asset at a pre-specified strike price $K$ only at the maturity date $T$.

    The payoffs at maturity are defined as:
    - Call: $C_T = \max(S_T - K, 0)$
    - Put:  $P_T = \max(K - S_T, 0)$

    Args:
        T (float): Time to maturity in years (e.g., 0.25 for 3 months).
        K (float): The strike price of the contract.
        option_type (str, optional): The type of option, either "call" or "put". 
            Defaults to "call".
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

