
# Option Pricing Library

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)  
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)   

A Python library for pricing **vanilla and exotic financial options**,  
with flexible market parameters and advanced numerical methods.


## Table of Contents

- [Installation](#installation)  
- [Global Parameters](#global-parameters)  
- [Supported Option Types & Methods](#supported-option-types--methods)  
- [Models](#models)  
- [Greeks](#greeks)  
- [Implied Volatility](#implied-volatility)  
- [Usage Examples](#usage-examples)  
  - [European Call (Black-Scholes)](#example-1-european-call-black-scholes)  
  - [Asian Option (Arithmetic average)](#example-2-asian-option-arithmetic-average)  
  - [Implied Volatility](#example-3-implied-volatility)  
- [Contributing](#contributing)  
- [License](#license) 

## Installation

```bash
git clone https://github.com/VincentCourtehoux/pricing_library.git
cd pricing_library
pip install -r requirements.txt
```

## Global Parameters

All models support customizable parameters:
- **Underlying price (S)**
- **Strike price (K)**
- **Maturity (T)**
- **Volatility (σ)**
- **Risk-free interest rate (r)**
- **Dividend yield (q, continuous dividend rate)**
- **Option type**: Call or Put

**Option-specific parameters** are also supported, e.g.:
- **average_type** for Asian options
- **barrier_level** for Barrier options
- **n_paths** and **n_steps** for Monte Carlo

## Supported Option Types & Methods

### Vanilla Options
- **European options**
  - Methods: Black-Scholes, Binomial Trees, Monte Carlo
- **American options**
  - Methods: Binomial Trees, Least Squares Monte Carlo (LSM)

### Exotic Options
- **Asian options**
  - Supports both arithmetic and geometric averages
  - Flexible monitoring dates 
  - Method: Monte Carlo
- **Barrier options**
  - Types: Up-and-Out, Down-and-Out, Up-and-In, Down-and-In
  - Methods: Monte Carlo and Analytical
- **Gap options**
  - Method: Monte Carlo and Analytical

## Models

- **Black-Scholes-Merton (BSM)**  
  Closed-form solution for European options.
  
- **Binomial Trees (Cox-Ross-Rubinstein)**  
  Flexible discrete-time model for European and American options.  

- **Monte Carlo Simulation**  
  General-purpose numerical method for European, Asian, Barrier and Gap options.  
  Simulates underlying paths using geometric Brownian motion.  
  Handles path-dependent features (averaging, barriers, triggers).  

- **Least Squares Monte Carlo (LSM)**  
  Extension of Monte Carlo for American options.  
  Uses regression on simulated paths to approximate early exercise value.  
  Regression can be polynomial or Laguerre basis functions.

## Greeks

The library supports numerical computation of Greeks for all option types:
- **Delta, Gamma, Vega, Theta, Rho**  
- Analytical formulas for Black-Scholes, numerical (finite differences) for Monte Carlo and Binomial models.

## Implied Volatility

Compute implied volatility from a market option price using **Newton-Raphson iteration**  
with analytical Vega from the Black-Scholes model.

## Usage Examples

### Example 1: European Call (Black-Scholes) 

```python
from pricing_library.core.pricing_service import PricingService
service = PricingService()

european_params = {
    'option_style': 'european',
    'S': 100,      # Underlying price
    'K': 100,      # Strike price
    'T': 1,        # Time to maturity in years
    'r': 0.05,     # Risk-free rate
    'sigma': 0.2,  # Volatility
    'q': 0.02,     # Dividend yield
    'option_type': 'call'
}

price = service.price_option(european_params, method='black_scholes')
print("European Call Price:", price)

greeks = service.calculate_greeks(european_params, method='black_scholes')
print("Greeks:", greeks)
```

### Example 2: Asian Option (Arithmetic average)

```python
from pricing_library.core.pricing_service import PricingService
service = PricingService()

asian_params = {
    'option_style': 'asian',
    'S': 100,
    'K': 100,
    'T': 1,
    'r': 0.05,
    'sigma': 0.2,
    'q': 0.02,
    'option_type': 'call',
    'average_type': 'arithmetic',
    'monitoring_dates': [0.25, 0.5, 0.75, 1.0]
}

price_asian = service.price_option(asian_params, method='monte_carlo')
print("Asian Call Price:", price_asian['price'])

greeks = service.calculate_greeks(asian_params, method='monte_carlo')
print("Delta:", greeks['delta'])
```

### Example 3: Implied Volatility

```python
from implied_volatility import ImpliedVolatility
iv = ImpliedVolatility

params = {
    'option_style': 'european',
    'price': 10.45,
    'S': 100,
    'K': 100,
    'T': 1,
    'r': 0.05,
    'q': 0.0,
    'option_type': 'call',
    'max_iterations': 100,
    'tolerance': 1e-6
}

implied_volatility = iv.calculate(params)
print("Implied Volatility:", implied_volatility)
```

## Contributing

Contributions are welcome!
Please fork the repo and submit a pull request.
- Follow PEP8 style
- Include tests for new features
- Update README with examples if needed

## License
This project is licensed under the MIT License.  
See the full license [here](https://opensource.org/licenses/MIT).
