import numpy as np 
import logging 
from quanteo.models.base_model import BaseModel

class GBM(BaseModel):
    """
    Geometric Brownian Motion (GBM) model for asset price dynamics.

    Simulates the continuous-time stochastic process commonly used in the 
    Black-Scholes-Merton framework. The asset price follows the stochastic 
    differential equation (SDE):

    $$dS_t = r S_t dt + \sigma S_t dW_t$$

    where $W_t$ is a standard Brownian motion.

    Args:
        S0 (float): The initial or current asset price.
        r (float): The annualized risk-free interest rate (e.g., 0.04 for 4%).
        sigma (float): The annualized volatility of the asset returns.
        t_time (float, optional): The current time in years. Defaults to 0.0.

    Raises:
        ValueError: If the volatility `sigma` is less than or equal to zero.
    """
    def __init__(self, S0: float, r: float, sigma: float, t_time: float = 0.0):
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