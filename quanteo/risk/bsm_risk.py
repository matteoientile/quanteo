import numpy as np
from scipy.stats import norm
from options.european import EuropeanOption
from models.gbm import GBM

class AnalyticalBSMGreeks:
    """
    Computes exact closed-form Greeks according to the Black-Scholes-Merton derivation.
    Strictly limited to European options under Geometric Brownian Motion.
    """
    def calculate(self, option, model) -> dict:
        """
        Returns a dictionary containing Delta, Gamma, Theta, Vega, and Rho.
        """
        if not isinstance(option, EuropeanOption):
            raise TypeError(f"BSM Greeks require a EuropeanOption. Got: {type(option).__name__}")
        if not isinstance(model, GBM):
            raise TypeError(f"BSM Greeks require a GBM model. Got: {type(model).__name__}")
        
        #time to maturity
        ttm = option.T - model.t_time

        if ttm <= 0:
            raise ValueError("Maturity T must be > current time t_time.")

        ttm_sroot = np.sqrt(ttm)

        #compute d1 and d2
        d1 = (np.log(model.S0 / option.K) + (model.r + 0.5 * model.sigma**2) * ttm) / (model.sigma * ttm_sroot)
        d2 = d1 - model.sigma * ttm_sroot

        #compute Gamma and Vega (same whether put or call)
        gamma = norm.pdf(d1)/(model.S0 * model.sigma * ttm_sroot)
        vega = model.S0 * norm.pdf(d1) * ttm_sroot

        #compute Delta, Theta and Rho
        if option.option_type == "call":
            delta = float(norm.cdf(d1))
            rho = float(option.K * ttm * np.exp(-model.r * ttm) * norm.cdf(d2))
            theta = float(-(model.S0 * norm.pdf(d1) * model.sigma) / (2 * ttm_sroot) - model.r * option.K * np.exp(-model.r * ttm) * norm.cdf(d2))
            
        elif option.option_type == "put":
            delta = float(norm.cdf(d1) - 1.0)
            rho = float(-option.K * ttm * np.exp(-model.r * ttm) * norm.cdf(-d2))
            theta = float(-(model.S0 * norm.pdf(d1) * model.sigma) / (2 * ttm_sroot) + model.r * option.K * np.exp(-model.r * ttm) * norm.cdf(-d2))

        return {
            "Delta": delta,
            "Gamma": gamma,
            "Vega": vega,
            "Theta": theta,
            "Rho": rho
        }