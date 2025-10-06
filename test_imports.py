from pricing_library.core.pricing_service import PricingService

service = PricingService()

params = {
    "option_style": "combination",
    "combination": "straddle",
    "S": 101,
    "K": 100,
    "T": 1,
    "r": 0.05,
    "sigma": 0.2,
}

result_greeks = service.calculate_greeks(params, method='black_scholes')
greeks = result_greeks['greeks']

print("\nStraddle Computation :")
print(f"Price : {greeks['price']:.4f}")
print(f"Delta : {greeks['delta']:.4f}")
print(f"Gamma : {greeks['gamma']:.6f}")
print(f"Vega  : {greeks['vega']:.4f}")
print(f"Theta : {greeks['theta']:.6f}")
print(f"Rho   : {greeks['rho']:.4f}")

params = {
    "option_style": "combination",
    "combination": "strangle",
    "S": 101,
    "K_call": 100,
    "K_put":100,
    "T": 1,
    "r": 0.05,
    "sigma": 0.2,
}

result_price = service.price_option(params, method='black_scholes')

result_greeks = service.calculate_greeks(params, method='black_scholes')
greeks = result_greeks['greeks']

print("\nStrangle Computation :")
print(f"Price : {greeks['price']:.4f}")
print(f"Delta : {greeks['delta']:.4f}")
print(f"Gamma : {greeks['gamma']:.6f}")
print(f"Vega  : {greeks['vega']:.4f}")
print(f"Theta : {greeks['theta']:.6f}")
print(f"Rho   : {greeks['rho']:.4f}")
