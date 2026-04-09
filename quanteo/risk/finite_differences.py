import numpy as np
from quanteo.risk.base_risk import BaseRisk
import copy

class FiniteDifferenceGreek(BaseRisk):
    """
    Computes option risk sensitivities (Greeks) using numerical finite differences.

    This class acts as a universal Greek calculator. It wraps any pricing engine 
    (e.g., Analytical, Monte Carlo, QMC) and approximates the partial derivatives of 
    the option price with respect to specific market parameters. 
    
    Central differences are utilized for symmetric parameters (Delta, Gamma, Vega, Rho) 
    to achieve higher order accuracy, while a forward difference is adopted for time 
    decay (Theta).

    Example (Central Difference for Delta):
    $$\Delta \approx \frac{V(S_0 + \Delta S) - V(S_0 - \Delta S)}{2 \Delta S}$$

    Args:
        pricer (object): An instantiated pricing engine (e.g., `MonteCarloPricer`). 
            The object must implement a standard `.price(option, model)` method.
        dS_percentage (float, optional): The fractional bump size for the underlying 
            asset price. Defaults to 0.01 (1% of S0).
        dsigma (float, optional): The absolute bump size for volatility. Defaults to 0.01.
        dr (float, optional): The absolute bump size for the risk-free rate. Defaults to 0.01.
        dttm (float, optional): The bump size for time to maturity, typically representing 
            one day in annualized terms. Defaults to 1/365.
    """
    def __init__(self, pricer, dS_percentage: float= 0.01, dsigma: float=0.01, dr: float=0.01, dttm: float=1/365):
        self.pricer = pricer 
        self.dS_percentage = dS_percentage
        self.dsigma = dsigma
        self.dr = dr
        self.dttm = dttm


    def greeks_calculator(self, option, model) -> dict:
        #store greeks
        greeks = {}

        #time to maturity
        ttm = option.T - model.t_time
        #current option price
        V0 = self.pricer.price(option, model).price # extract the price, since pricer.price return a PricingResult object 

        #compute S0 + dS and S0 - dS
        dS = model.S0 * self.dS_percentage
        S_fw = model.S0 + dS
        S_bw = model.S0 - dS

        # Rather than repeating the simulation_path and payoff structure with little changes
        # , copy the method and change only the necessary parameters
        model_fw = copy.deepcopy(model)
        model_bw = copy.deepcopy(model)
        #update S0 --> S0+dS and S0 - dS
        model_fw.S0 = S_fw
        model_bw.S0 = S_bw
        
        #option payoffs if S0 = S0+ds and S0 = S0 - dS
        V0_fw = self.pricer.price(option, model_fw).price
        V0_bw = self.pricer.price(option, model_bw).price

        #1 & 2. Compute Delta & Gamma: dV/dS and d^V/dS^2 
        greeks["Delta"] = (V0_fw - V0_bw)/(2*dS)
        greeks["Gamma"] = (V0_fw- 2*V0 + V0_bw)/(dS**2)

        #3. Compute Vega: dV/dsigma
        #some methods pricing methods do not belong to sigma, therefore check if sigma is actually passed through model
        if hasattr(model, "sigma"):
            model_fw = copy.deepcopy(model)
            model_bw = copy.deepcopy(model)
            #update sigma --> sigma+dsigma and sigma - dsigma
            model_fw.sigma = model.sigma + self.dsigma
            model_bw.sigma = model.sigma - self.dsigma

            V0_fw = self.pricer.price(option, model_fw).price
            V0_bw = self.pricer.price(option, model_bw).price

            greeks["Vega"] = (V0_fw - V0_bw)/(2*self.dsigma)
        else:
            greeks["Vega"] = None 

        
        #4. Compute rho
        model_fw = copy.deepcopy(model)
        model_bw = copy.deepcopy(model)
        #update r --> r+dr and r - dr
        model_fw.r = model.r + self.dr
        model_bw.r = model.r - self.dr

        V0_fw = self.pricer.price(option, model_fw).price
        V0_bw = self.pricer.price(option, model_bw).price

        greeks["Rho"] = (V0_fw - V0_bw)/(2*self.dr)


        #5. Compute Theta
        model_fw = copy.deepcopy(model)
        model_fw.t_time = model.t_time + self.dttm
        if model_fw.t_time < option.T:
            V0_fw = self.pricer.price(option, model_fw).price
            greeks["Theta"] = (V0_fw - V0) / self.dttm 
        else:
            greeks["Theta"] = 0.0

        return greeks




