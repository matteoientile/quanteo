import numpy as np
from scipy.stats import norm
from quanteo.pricers.base_pricer import BasePricer, PricingResult

class BSMPricer(BasePricer):
    """
    Analytical pricing engine for vanilla European options using the 
    Black-Scholes-Merton (BSM) framework.

    This pricer provides a closed-form solution for options that can only be 
    exercised at maturity, assuming the underlying asset follows Geometric 
    Brownian Motion with constant risk-free rates and volatility.

    The price $V$ is determined by:
    - Call: $S_0 N(d_1) - K e^{-rT} N(d_2)$
    - Put:  $K e^{-rT} N(-d_2) - S_0 N(-d_1)$

    where:
    $$d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma \sqrt{T}}$$
    $$d_2 = d_1 - \sigma \sqrt{T}$$

    Args:
        option (EuropeanOption): The contract definition.
        model (GBM): The market model containing $S_0, r, \sigma$.

    Returns:
        PricingResult: An object containing the analytical price.

    Raises:
        ValueError: If the time to maturity is negative.
    """
    def price(self, option, model) -> PricingResult:
        """
        Returns the exact closed-form Black-Scholes-Merton price.
        """
        # check syntax
        ttm = option.T - model.t_time
        if ttm == 0:
            # We wrap S0 in a numpy array because the option.payoff expects one
            value = float(option.payoff(np.array([model.S0]))[0])
            return PricingResult(price=value)
        if ttm < 0:
            raise ValueError("Option has already expired.")

        # compute d1 and d2
        d1 = (np.log(model.S0/option.K) + (model.r + model.sigma**2/2)*ttm) / (model.sigma* np.sqrt(ttm)) # d1, equal both for put and call. Look at BSM derivation for more info
        d2 = d1 - model.sigma * np.sqrt(ttm) # d2, equal both for put and call. Look at BSM derivation for more info
        
        if option.option_type == "call":
            value = float(model.S0 * norm.cdf(d1) - option.K * np.exp(-model.r * ttm) * norm.cdf(d2))
            return PricingResult(price=value)
        elif option.option_type == "put":
            value = float(option.K * np.exp(-model.r*ttm) * norm.cdf(-d2) - model.S0 * norm.cdf(-d1))
            return PricingResult(price=value)



class GeometricAsianPricer(BasePricer):
    """
    Analytical pricing engine for discrete Geometric Asian Options.

    While Arithmetic Asian options lack a closed-form solution, Geometric Asian 
    options can be priced analytically because the product of log-normal 
    variables is itself log-normal. This pricer calculates the exact expected 
    value by adjusting the drift and volatility parameters to account for the 
    averaging process.

    This engine is the mathematical 'anchor' for the Control Variate method.

    The adjusted parameters used for the Black-Scholes-like formula are:
    - $a$: The adjusted log-drift.
    - $b$: The adjusted variance over the life of the option.

    Args:
        q (float, optional): The continuous dividend yield of the underlying asset. 
            Defaults to 0.0.

    Returns:
        PricingResult: The exact analytical price of the Geometric Asian contract.
    """
    def __init__(self, q: float=0.0):
        self.q = q 

    def price(self, option, model):
        #parameters
        T = option.T
        K = option.K
        N = option.N
        S0 = model.S0
        r = model.r
        t_time = model.t_time
        sigma = model.sigma
        ttm = T-t_time

        # ASSUMPTION: m = 0 --> the last time at which we observed the price of the underlying asset corresponds the time 0, i.e. price = S0
        dt = ttm/N
        nu = r - sigma**2/2 - self.q
        a = np.log(S0) + nu*dt + 0.5*nu*(ttm - dt)
        b = sigma**2 *dt + sigma**2 * (ttm - dt) * (2*N - 1)/(6*N)
        x = ( a - np.log(K) + b)/np.sqrt(b) 

        if option.option_type == "call":
            value = np.exp(-r*ttm) * (np.exp(a + b / 2) * norm.cdf(x) - K * norm.cdf(x - np.sqrt(b)))
        elif option.option_type == "put":
            # Using standard symmetry for the put option
            value = np.exp(-r * ttm) * (K * norm.cdf(-x + np.sqrt(b)) - np.exp(a + 0.5 * b) * norm.cdf(-x))
        else:
            raise ValueError(f"Invalid option type: {option.option_type}")

        return PricingResult(price=float(value))
