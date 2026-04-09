import numpy as np
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract base class for the simulation of underlying asse paths.
    A model object must contain the common parameters to all the models.
    """
    def __init__(self, S0: float, r: float, t_time: float = 0.0):
        """
        Constructor.
        Args:
            S0 (float): Current asset price. If t_time = 0 then S0 = asset price when the option contract is emitted
            r (float): Risk free rate (annualized)
            t_time (float, optional): Current time, time elapsed since option emission. Defaults to 0.0, i.e. at the moment of option contract emission.
        """
        self.S0 = S0
        self.r = r 
        self.t_time = t_time 

        #check syntax
        if self.S0 <= 0:
            raise ValueError("Current asset price S0 must be > 0")

    @abstractmethod
    def simulate_paths(self, T: float, n_paths: int, n_steps: int, seed: int=42) -> np.ndarray:
        """_summary_

        Args:
            T (float): Maturity time (since the emission date). 
            n_paths (int): Number of asset trajectories
            n_steps (int): Number of time steps between t_time and T
            seed (int, optional): Seed for reproducibility. Defaults to 42.

        Returns:
            np.ndarray: 2D array (n_paths, n_steps + 1) containing the price trajectories
        """
        pass
