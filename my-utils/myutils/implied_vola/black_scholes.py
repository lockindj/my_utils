# myutils/implied_vola/black_scholes.py
from __future__ import annotations
from math import log, sqrt, exp
from typing import Union
from myutils.math_methods import norm_cdf, norm_pdf

Number = Union[float, int]

def zero_volatility_limit(S: Number, K: Number, T: Number, r: Number, q: Number, is_call: bool) -> float:
    """
    Limit price as sigma -> 0 (deterministic forward price development).
    """
    fwd = S * exp(-q * T) - K * exp(-r * T)
    if is_call:
        return max(fwd, 0.0)
    else:
        return max(-fwd, 0.0)

def bs_price(S: Number, K: Number, T: Number, r: Number, q: Number, sigma: Number, is_call: bool) -> float:
    """
    Black–Scholes price with continuous compounding for rates and dividend yield.

    Parameters
    ----------
    S : spot price (must be > 0)
    K : strike (must be > 0)
    T : time to maturity in YEARS (e.g., DTE/365.0)
    r : risk-free rate as DECIMAL (0.02 for 2%), continuously compounded
    q : dividend yield as DECIMAL, continuously compounded
    sigma : volatility as DECIMAL (e.g., 0.2 for 20%)
    is_call : True for call, False for put

    Returns
    -------
    float: theoretical Black–Scholes price
    
    Notes
    -----
    - At expiry (T == 0), returns intrinsic value: max(S-K, 0) for calls,
      max(K-S, 0) for puts.
    - Rates and dividend yield are assumed to be continuously compounded.
    """
    
    if S <= 0 or K <= 0:
        raise ValueError("S (spot price) and K (strike) must be positive.")
    if T <= 0:
        # At expiry, price is intrinsic value
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    if sigma <= 0:
        # Zero-volatility limit equals discounted intrinsic on forward
        return zero_volatility_limit(S, K, T, r, q, is_call)

    st = sqrt(T)
    d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * st)
    d2 = d1 - sigma * st

    if is_call:
        return S * exp(-q * T) * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)
    else:
        return K * exp(-r * T) * norm_cdf(-d2) - S * exp(-q * T) * norm_cdf(-d1)

def vega(S: Number, K: Number, T: Number, r: Number, q: Number, sigma: Number) -> float:
    """
    Black–Scholes Vega (∂Price/∂sigma). Units: price per 1.0 change in sigma.
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    st = sqrt(T)
    d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * st)
    return S * exp(-q * T) * norm_pdf(d1) * st
