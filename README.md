# QUANTEO: Quantitative Options Pricing & Risk

> Quanteo is an Object-Oriented Python library for derivative pricing and risk sensitivity analysis. This project is developed as part of my learning journey during the **MSc in Mathematical Engineering** at **Politecnico di Torino**. The library implements stochastic models and variance reduction techniques to estimate option prices and Greeks, particularly for path-dependent derivatives.

---

**Author:** Matteo Ientile  
*MSc Mathematical Engineering – Politecnico di Torino*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/matteo-ientile/)
[![X](https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/mttntl02)
---

## Installation

You can install the official release directly via pip:
```bash
pip install quanteo
```

## Functionalities available at the moment

### 1. Simulation & Pricing Engines
* **Monte Carlo:** Path generation for Geometric Brownian Motion (GBM).
* **Quasi-Monte Carlo:** Sobol low-discrepancy sequences for improved convergence.
* **Variance Reduction:** Control Variates (anchored to analytical solutions) and Antithetic Sampling.
* **Analytical Models:** Closed-form Black-Scholes-Merton (BSM) for European and Geometric Asian options.

### 2. Risk Management (Greeks)
* Computation of Δ (Delta), Γ (Gamma), ν (Vega), ρ (Rho), and Θ (Theta).
* **Finite Differences:** Central difference approximations for numerical sensitivities.
* **CRN (Common Random Numbers):** Epsilon caching to ensure stability in numerical derivatives.

### 3. Contract Types
* **European Options:** Standard Call/Put payoffs.
* **Asian Options:** Discrete Arithmetic and Geometric averages.

---

## Examples
The `examples/` directory contains notebooks demonstrating the implementation:
* `european_options_and_greeks.ipynb`: Benchmarking BSM, MC, and QMC engines.
* `asian_options_and_variance_reduction.ipynb`: Statistical impact of Control Variates on Asian contracts.

---

## Usage

```python
from quanteo.models import GBM
from quanteo.options import ArithmeticAsianOption, GeometricAsianOption
from quanteo.pricers import MonteCarloPricer, ControlVariateMC, GeometricAsianPricer

# 1. Market Data
model = GBM(S0=85.0, r=0.04, sigma=0.42)

# 2. Options (126 observations)
target = ArithmeticAsianOption(T=0.5, K=88.0, N=126)
control = GeometricAsianOption(T=0.5, K=88.0, N=126)

# 3. Analytical Anchor
exact_cv = GeometricAsianPricer().price(control, model).price

# 4. Monte Carlo with Control Variate
mc = MonteCarloPricer(n_paths=5000, n_steps=126)
cv_engine = ControlVariateMC(mc)

result = cv_engine.price(target, control, exact_cv, model)

print(f"Price: {result.price:.4f}")
print(f"95% CI: [{result.metrics['confidence_interval'][0]:.4f}, {result.metrics['confidence_interval'][1]:.4f}]")
```

---

## Roadmap
* [ ] **Heston Model:** Stochastic volatility integration.
* [ ] **Jump Diffusion:** Merton model for discontinuous price paths.
* [ ] **American Options:** Longstaff-Schwartz (LSM) implementation.
* [ ] **Dashboard:** Streamlit interface for visualization.
