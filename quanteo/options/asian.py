import numpy as np 
from quanteo.options.base_option import BaseOption

class ArithmeticAsianOption(BaseOption):
    """
    Contract definition for an Arithmetic Asian Option.

    An Asian option is a path-dependent exotic derivative where the payoff is 
    determined by the arithmetic average of the underlying asset's price over a 
    pre-determined set of observation periods, rather than just the final price at maturity.

    The arithmetic average $A$ is defined as:
    $$A = \frac{1}{N} \sum_{i=1}^{N} S_{t_i}$$

    The payoffs at maturity $T$ are:
    - Call: $\max(A - K, 0)$
    - Put:  $\max(K - A, 0)$

    Args:
        T (float): Option maturity in years (e.g., 0.5 for 6 months).
        K (float): The strike price of the contract.
        N (int): The number of discrete price observations used to compute the 
            arithmetic average. Note that $S_0$ is typically excluded from this average.
        option_type (str, optional): The type of option, either "call" or "put". 
            Defaults to "call".
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
    Contract definition for a Geometric Asian Option.

    An Asian option is a path-dependent exotic derivative where the payoff is 
    determined by the geometric average of the underlying asset's price over a 
    pre-determined set of observation periods. 
    
    Because the product of log-normally distributed variables (like stock prices 
    in the BSM model) is also log-normally distributed, Geometric Asian options 
    possess a closed-form analytical solution, making them ideal Control Variates 
    for pricing Arithmetic Asian options.

    The geometric average $G$ is defined as:
    $$G = \left( \prod_{i=1}^{N} S_{t_i} \right)^{\frac{1}{N}}$$

    The payoffs at maturity $T$ are:
    - Call: $\max(G - K, 0)$
    - Put:  $\max(K - G, 0)$

    Args:
        T (float): Option maturity in years (e.g., 0.5 for 6 months).
        K (float): The strike price of the contract.
        N (int): The number of discrete price observations used to compute the 
            geometric average. Note that $S_0$ is typically excluded from this average.
        option_type (str, optional): The type of option, either "call" or "put". 
            Defaults to "call".
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
