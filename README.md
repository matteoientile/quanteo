# Quantitative Options Pricing & Risk Library 

> **Status:** 🚧 *Active Development:* A scalable, Object-Oriented Python library for derivative pricing, risk management, and quantitative stochastic modeling. 

Built to handle massive volatility environments, this library bridges the gap between closed-form analytical mathematics and high-performance computational simulation. It features an optimized Monte Carlo engine with advanced variance reduction techniques, capable of shrinking confidence intervals for path-dependent derivatives without brute-forcing millions of trajectories.

## Core Features
* **Modular OOP Architecture:** Clean inheritance trees (`BaseOption`, `BasePricer`) ensuring strict separation between contract definitions and mathematical pricing models.
* **Analytical Engines:** Exact Black-Scholes-Merton (BSM) closed-form pricing.
* **Risk Sensitivities:** Greek computations ($\Delta, \Gamma, \mathcal{V}, \rho, \Theta$) utilizing Finite Difference methods alongside CRN (Common Random Numbers).
* **Advanced Monte Carlo Simulation:** Highly optimized path generation using Geometric Brownian Motion (GBM).
* **Variance Reduction Techniques:** Control Variates (anchored to analytical bounds) and Antithetic Sampling to exponentially reduce standard error. 
* **Path-Dependent Derivatives:** Support for discrete Arithmetic and Geometric Asian Options.

## Coming Soon
* [ ] **Advanced models for the underlying asset price:** Heston, Merton and many other models
* [ ] **New options**: Americans, Exotics, Barriers...

And much more.


## Quick Start
*(Example: Pricing an Arithmetic Asian Option using a Geometric Control Variate)*

```python
from options import ArithmeticAsianOption, GeometricAsianOption
from pricers import MonteCarloPricer, ControlVariateMC, GeometricAsianPricer
from models import GBM

# 1. Define the Market & Asset
model = GBM(S0=85.0, r=0.04, sigma=0.42, t_time=0.0)

# 2. Define the Contracts (126 observation days)
target_option = ArithmeticAsianOption(T=0.5, K=88.0, N=126, option_type="call")
cv_option = GeometricAsianOption(T=0.5, K=88.0, N=126, option_type="call")

# 3. Anchor to the Exact Analytical Price
exact_cv_price = GeometricAsianPricer().price(option=cv_option, model=model).price

# 4. Execute Variance Reduction Monte Carlo
mc_engine = MonteCarloPricer(n_paths=5000, n_steps=126)
cv_engine = ControlVariateMC(mc_pricer=mc_engine)

result = cv_engine.price(
    option=target_option, 
    control_var=cv_option, 
    exact_cv_price=exact_cv_price, 
    model=model
)

print(f"Option Price: {result.price}")
print(f"Optimal Multiplier (c*): {result.metrics['c_star']}")

