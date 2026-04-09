import numpy as np
from quanteo.pricers.base_pricer import PricingResult
from quanteo.options.sum_prices import SumPricesCV
import logging

class ControlVariateMC:
    """
    Implements Monte Carlo simulation with the very well 
    effective Variance Reduction method, known as "Control Variates".
    
    We avoid to create again the Monte Carlo logic by calling the existing one.
    """
    def __init__(self, mc_pricer):
        # The user previously called Monte Carlo as its Option pricer. 
        # Here we recall its logic by passing the object already instantiated from the user.
        self.pricer = mc_pricer

        if self.pricer.antithetic:
            logging.warning(f"You are overlapping Antithetic Sampling and Control Variates: this may lead to silent errors in estimates")

    #======================================
    #========= PRIVATE METHODS ============
    #======================================
    def _pilot_replications(self, option, control_var, model, n_pilot: int) -> float:
        """
        Estimates Var[Y] and Cov(X, Y) to compute c*, 
        using an independent pilot replication to maintain an unbiased estimator.
        """
        original_n = self.pricer.n_paths
        self.pricer.n_paths = n_pilot

        #generate pseudorandom numbers
        self.pricer._cached_epsilon = None
        self.pricer._generate_epsilon()

        #generate n_pilot paths
        pilot_paths = model.simulate_paths(
            T = option.T,
            n_paths = n_pilot,
            n_steps = self.pricer.n_steps,
            epsilon = self.pricer._cached_epsilon
        )

        #compute discounted payoffs
        ttm = option.T - model.t_time
        discount = np.exp(-model.r * ttm)
        X_payoffs_pilot = option.payoff(paths=pilot_paths)*discount # discounted payoffs for target option
        Y_payoffs_pilot = control_var.payoff(paths=pilot_paths)*discount #discounte payoffs based on control variates option. 

        #compute c*
        cov_matrix =np.cov(X_payoffs_pilot, Y_payoffs_pilot)
        var_y = cov_matrix[1, 1]
        if var_y != 0:
            c_star = cov_matrix[0,1]/var_y
        else:
            c_star = 0.0

        self.pricer.n_paths = original_n

        return c_star

    #======================================
    #========= PUBLIC METHODS =============
    #======================================

    def price(self, option, control_var, exact_cv_price: float, model, n_pilot: int = 1000) -> PricingResult:
        """
        Prices the target option using a Control Variate Variance Reduction technique.
        It runs a two-phase simulation (Pilot and Main) to guarantee an unbiased estimator.

        Args:
            option (object): The target option to be priced (e.g., ArithmeticAsianOption). Must have a .payoff(paths) method.
            control_var (object): The control variate proxy (e.g., GeometricAsianOption or SumPricesCV). Must have a .payoff(paths) method.
            exact_cv_price (float): The exact, analytical expected value of the control variate.
            model (object): The underlying asset model (e.g., GBM) used to simulate trajectories.
            n_pilot (int, optional): The number of paths to simulate in the pilot run to estimate c*. Defaults to 1000.

        Returns:
            PricingResult: A data carrier containing the variance-reduced price and a metrics dictionary (Confidence Interval and optimal c*).
        """

        c_star = self._pilot_replications(option, control_var, model, n_pilot)

        #generate pseudorandom numbers
        self.pricer._cached_epsilon = None
        self.pricer._generate_epsilon()

        #generate paths
        paths = model.simulate_paths(
            T = option.T,
            n_paths = self.pricer.n_paths,
            n_steps = self.pricer.n_steps,
            epsilon = self.pricer._cached_epsilon
        )

        #compute payoffs (X and Y)
        ttm = option.T - model.t_time 
        discount = np.exp(-model.r * ttm)
        X_payoffs = option.payoff(paths)*discount
        Y_payoffs = control_var.payoff(paths)*discount

        #compute actual payoffs with X_C = X + c* \cdot (Y - nu)
        cv_payoffs = X_payoffs - c_star*(Y_payoffs - exact_cv_price)
        #compute average payoff
        avg_payoff = float(np.mean(cv_payoffs))

        #confidence intervals
        ci = self.pricer._compute_confidence_intervals(
            discounted_payoffs=cv_payoffs,
            estimated_value=avg_payoff
        )

        return PricingResult(
            price=avg_payoff,
            metrics=
            {
                "confidence_interval" : ci,
                "c_star" : c_star
            }
        )