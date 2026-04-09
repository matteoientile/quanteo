import numpy as np 
import logging 
from quanteo.models.base_model import BaseModel

class GBM(BaseModel):
    """
    Asset price modelled according to Geometric Brownian Motion (GBM).
    """
    def __init__(self, S0: float, r: float, sigma: float, t_time: float = 0.0):
        """
        Constructor. All the parameters needed to perform the simulation must be passed

        Args:
            S0 (float): Current asset price. If t_time = 0 then S0 = asset price when the option contract is emitted
            r (float): Risk free rate (annualized)
            sigma (float): Volatility
            t_time (float, optional): Current time. Defaults to 0.0, i.e. at the moment of option contract emission.
        """
        super().__init__(S0, r, t_time)
        # sigma must be added, base_model does not contain it since sigma is not required for every model. 
        self.sigma = sigma

        # check syntax
        if self.sigma <= 0:
            raise ValueError("Volatility (sigma) must be > 0.")

    
    def simulate_paths(self, T: float, n_paths: int, n_steps: int, epsilon: np.ndarray) -> np.ndarray:
        """
        Method that simulates asset trajectories according to GBM.
        epsilon array must be previously generated (within pricer)
        """
        # check syntax
        if T <= self.t_time:
            raise ValueError("It must be T (maturity) >= t_time (current time, time elapsed since contract emission).")

        #step size
        dt = (T - self.t_time)/n_steps

        #initialize void array to fill
        paths = np.zeros(shape=(n_paths, n_steps+1))
        paths[:, 0] = self.S0
            

        # S_{t+1} = S_{t} * exp{ (r - sigma^2/2)*dt + sigma*sqrt{dt}*epsilon}, epsilon \sim N(0, 1):

        #compute constant terms
        drift_term = (self.r - 0.5 * self.sigma**2)*dt 
        vol_term = self.sigma * np.sqrt(dt)

        # variation term
        variation_term = np.exp(drift_term + vol_term*epsilon)
        #paths
        paths[:, 1:] = self.S0 * np.cumprod(variation_term, axis=1) 

        return paths