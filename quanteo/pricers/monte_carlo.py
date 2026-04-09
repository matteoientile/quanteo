import numpy as np
from scipy.stats import t
from quanteo.pricers.base_pricer import BasePricer, PricingResult

class MonteCarloPricer(BasePricer):
    """
    Standard Monte Carlo pricing engine for path-dependent and vanilla options.

    This engine simulates multiple asset price trajectories under the risk-neutral 
    measure. The option price is estimated as the discounted expected value of the 
    payoffs across all simulated paths.

    Features:
    - **Antithetic Variates**: Optional variance reduction by simulating 
      mirrored Brownian paths ($W_t$ and $-W_t$).
    - **CRN Caching**: Caches the standard normal random matrix (epsilon) to 
      ensure Common Random Numbers are used during sensitivity analysis, 
      eliminating 'simulation noise' from Greek calculations.
    - **Confidence Intervals**: Calculates the statistical error of the estimate 
      based on the Student's t-distribution.

    The estimator converges to the true price at a rate of $\mathcal{O}(1/\sqrt{N})$.

    Args:
        n_paths (int, optional): Number of price trajectories to simulate. 
            Defaults to 10000.
        n_steps (int, optional): Number of time steps per path. Defaults to 1.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        alpha (float, optional): Confidence level for the error margin. 
            Defaults to 0.95 (95% CI).
        antithetic (bool, optional): Whether to use Antithetic Sampling. 
            Defaults to False.
    """
    def __init__(self, n_paths: int = 10000, n_steps: int=1, seed: int = 42, alpha: float = 0.95, antithetic: bool = False):
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed
        self.alpha = alpha
        self.antithetic = antithetic
        self._cached_epsilon = None
    
    #======================================
    #========= PRIVATE METHODS ============
    #======================================
    def _generate_epsilon(self):
        """
        Generates and caches the random numbers if they don't already exist
        """
         # generate epsilon if None is passed
        if self._cached_epsilon is None:
            #random generator
            rng = np.random.default_rng(self.seed)
            #generate random numbers
            if self.antithetic:
                #check syntax
                if self.n_paths % 2 != 0:
                    raise ValueError(f"n_paths must be even when antithetic=True. Instead {self.n_paths} passed")

                epsilon_half = rng.standard_normal(size=(self.n_paths//2, self.n_steps)) #generate X_1(1), X_1(2), ...
                self._cached_epsilon = np.vstack((epsilon_half, -epsilon_half)) # couple the get negatively correlted samples (according to Antitethic Sampling) 

            else:
                self._cached_epsilon = rng.standard_normal(size=(self.n_paths, self.n_steps))

        return self._cached_epsilon


    def _compute_confidence_intervals(self, discounted_payoffs: np.ndarray, estimated_value: float):
        """
        Private method to compute confidence intervals
        Args:
            - discounted_payoffs (ndarray): payoffs with discount exp{-r*ttm} already applied
            - estimated_value (float): option value estimated by the method chosen

        Returns:
            - confidence_intervals (tuple[float, float]): return a tuple [lower_bound, upper_bound] 
        """
        # CI
        n = len(discounted_payoffs)
        std_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(n)
        confidence_intervals = t.interval( 
            confidence=self.alpha,
            df = n-1,
            loc = estimated_value,
            scale=std_error
        )
        return confidence_intervals


    #======================================
    #========= PUBLIC METHODS =============
    #======================================

    def price(self, option, model) -> PricingResult:
        """
        Perform Monte Carlo Simulation

        Args:
            option (object): option object, able to compute the payoff
            model (object): model object, able to compute trajectories
        Returns:
            PricingResult: A data carrier containing the average payoff (float) and a metrics dictionary with the Confidence Interval boundaries.
        """
        epsilon = self._generate_epsilon()

        # Simulate paths with the model chosen
        paths = model.simulate_paths(
            T = option.T,
            n_paths = self.n_paths,
            n_steps = self.n_steps,
            epsilon=epsilon
        )

        # Compute array of payoffs
        payoffs = option.payoff(paths) #return an array containing the corresponding payoff for each trajectory

        # Discounted the payoffs
        ttm = option.T - model.t_time
        discounted_payoffs = np.exp(-model.r * ttm)*payoffs 

        if self.antithetic:
            #perform antithetic sampling properly
            n_half = self.n_paths // 2
            paired_payoffs = (discounted_payoffs[:n_half] + discounted_payoffs[n_half:]) / 2.0

            avg_payoff = float(np.mean(paired_payoffs))
            ci = self._compute_confidence_intervals(paired_payoffs, avg_payoff)
        else:
            # Compute average payoff and confidence intervals
            avg_payoff = float(np.mean(discounted_payoffs))
            ci = self._compute_confidence_intervals(discounted_payoffs, avg_payoff)

        return PricingResult(
                price=avg_payoff,
                metrics={"confidence interval" : ci}
            )
        