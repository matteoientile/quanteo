import numpy as np
from quanteo.pricers.base_pricer import BasePricer, PricingResult
from scipy.stats import norm, qmc
import math
import logging

class QuasiMCPricer(BasePricer):
    #put docstring 
    def __init__(self, n_paths:int=128, n_steps: int = 1, seed: int=42):
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed

    #======================================
    #========= PUBLIC METHODS ============
    #======================================

    def price(self, option, model) -> PricingResult:
        """
        Method that, taken all the inputs required, returns the option (put or call) price
        according to the QUASI-Monte Carlo Simulation approach.
        This function uses the Sobol sequences, a safe and standard way to generate low discrepancy sequences.
        Scrambling is internally forced in order to enhance stability.  

        Args:
            option (object): option object, able to compute the payoff
            model (object): model object, able to compute trajectories
        Returns:
            - float : estimated option value
        """
        # Generate Sobol sequence
        m = math.ceil(math.log2(self.n_paths)) #round to the nearest integer
        actual_paths = 2**m
        
        # Make the user aware of the actual number of paths
        if actual_paths != self.n_paths:
            logging.warning(f"n_paths ({self.n_paths}) is not a power of 2. Adjusted to {actual_paths} to respect Sobol sequence requirements.")

        # Only the last price matters -> 1 step -> d=1
        sampler = qmc.Sobol(d=self.n_steps, scramble=True, seed=self.seed) # define the sequence type (we choose Sobol, however some other choiches available here: https://docs.scipy.org/doc/scipy/reference/stats.qmc.html)
        seq = sampler.random_base2(m=m)
        
        # From Sobol sequence \sim Uniform -> Z \sim N(0,1)
        epsilon = norm.ppf(seq)

        # Asset trajectories with Sobol sequences
        paths = model.simulate_paths(
            T = option.T,
            n_paths = actual_paths,
            n_steps = self.n_steps, 
            epsilon=epsilon
        )

        # Compute payoffs array
        payoffs = option.payoff(paths)

        # Discount payoff
        ttm = option.T - model.t_time
        discounted_payoffs = np.exp(-model.r*ttm)*payoffs

        # Compute average payoff 

        return PricingResult(price=float( np.mean(discounted_payoffs) ))