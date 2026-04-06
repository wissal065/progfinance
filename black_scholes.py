import numpy as np
from scipy.stats import norm

# Calcul de d1 et d2
def d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

# Prix du call
def call_price(S, K, T, r, sigma):
    return S * norm.cdf(d1(S, K, T, r, sigma)) - K * np.exp(-r * T) * norm.cdf(d2(S, K, T, r, sigma))

# Prix du put
def put_price(S, K, T, r, sigma):
    return K * np.exp(-r * T) * norm.cdf(-d2(S, K, T, r, sigma)) - S * norm.cdf(-d1(S, K, T, r, sigma))

if __name__ == "__main__":
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.2

    print("Call price:", call_price(S, K, T, r, sigma))
    print("Put price:", put_price(S, K, T, r, sigma))