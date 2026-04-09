import numpy as np
from quanteo.risk.base_risk import BaseRisk
import copy

class FiniteDifferenceGreek(BaseRisk):
    """
    Compute Greeks for any Option with Symmetric (Central) Finite Differences numerical method.
    Takes a pricer method to simulate paths, then perform FD. 
    For the time dependet greek, theta, forward method is adopted rather than FD. 
    """
    def __init__(self, pricer, dS_percentage: float= 0.01, dsigma: float=0.01, dr: float=0.01, dttm: float=1/365):
        """
        Constructor.
        - pricer (obj): pricer object that implements Monte Carlo, Quasi-MC, ... 
        - dS_percentage (float): size of dS w.r.t. S0. Default is 0.01 (1%)
        """
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




