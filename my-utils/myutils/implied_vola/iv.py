# myutils/implied_vola/iv.py
from __future__ import annotations

import math
from math import exp
from typing import Tuple

from myutils.implied_vola.black_scholes import bs_price, vega
from myutils.math_methods.numerical_solvers.Newton_Raphson import newton_basic
from myutils.math_methods.numerical_solvers.Bisection import bisection_basic


def _no_arb_bounds(
    S: float, K: float, T: float, r: float, q: float, is_call: bool
) -> Tuple[float, float]:
    """
    No-arbitrage price bounds under continuous compounding.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to maturity in YEARS.
    r : float
        Risk-free rate (decimal, continuous).
    q : float
        Dividend yield (decimal, continuous).
    is_call : bool
        True for call, False for put.

    Returns
    -------
    (lower, upper) : tuple of float
        Lower/upper price bounds for the option.
    """
    disc_r = exp(-r * T)
    disc_q = exp(-q * T)
    if is_call:
        lower = max(S * disc_q - K * disc_r, 0.0)
        upper = S * disc_q
    else:
        lower = max(K * disc_r - S * disc_q, 0.0)
        upper = K * disc_r
    return lower, upper


def implied_vol_bs(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    is_call: bool,
    *,
    sigma_init: float = 0.2,
    tol: float = 1e-6,
    max_iter: int = 100,
    bounds: Tuple[float, float] = (1e-6, 5.0),
    fallback_to_bisection: bool = True,
) -> Tuple[float, str, str]:
    """
    Black–Scholes implied volatility via Newton–Raphson with bisection fallback.

    Parameters
    ----------
    price : float
        Observed option price (e.g., mid, bid, or ask).
    S : float
        Spot price (must be > 0).
    K : float
        Strike price (must be > 0).
    T : float
        Time to maturity in YEARS (must be > 0).
    r : float
        Risk-free rate as DECIMAL, continuously compounded.
    q : float
        Dividend yield as DECIMAL, continuously compounded.
    is_call : bool
        True for call, False for put.
    sigma_init : float, default 0.2
        Initial guess for volatility.
    tol : float, default 1e-6
        Convergence tolerance for the solvers.
    max_iter : int, default 100
        Maximum iterations per solver.
    bounds : (float, float), default (1e-6, 5.0)
        Lower/upper bounds for sigma.
    fallback_to_bisection : bool, default True
        If True, attempts bisection when Newton does not converge.

    Returns
    -------
    tuple
        (sigma, method, reason)
        - sigma (float): implied volatility (NaN on failure).
        - method (str): 'newton', 'bisection', or 'failed'.
        - reason (str): 'ok' on success, otherwise a short failure reason
          (e.g., 'price_out_of_bounds', 'no_bracket', 'max_iter', 'T_le_zero').

    Notes
    -----
    - Validates the observed price against no-arbitrage bounds before solving.
    - Uses continuous compounding for r and q.
    """
    # Basic input checks
    if S <= 0 or K <= 0:
        return (math.nan, "failed", "bad_inputs")
    if T <= 0:
        # At expiry, IV is not defined (payoff only).
        return (math.nan, "failed", "T_le_zero")

    # No-arbitrage price check
    lb, ub = _no_arb_bounds(S, K, T, r, q, is_call)
    eps = 1e-10
    if price < lb - eps or price > ub + eps:
        return (math.nan, "failed", "price_out_of_bounds")

    # Define residual and derivative for Newton: f(sigma) = BS(sigma) - price
    def f(sig: float) -> float:
        return bs_price(S, K, T, r, q, sig, is_call) - price

    def fp(sig: float) -> float:
        return vega(S, K, T, r, q, sig)

    # Try Newton first
    nres = newton_basic(f, fp, x0=sigma_init, tol=tol, max_iter=max_iter, bounds=bounds)
    if nres.get("converged", False):
        return (float(nres["root"]), "newton", "ok")

    if not fallback_to_bisection:
        return (math.nan, "failed", nres.get("reason", "no_convergence"))

    # Prepare a valid bracket for bisection
    lo, hi = bounds
    flo, fhi = f(lo), f(hi)

    # If no sign change, try extending the upper bound once or twice
    if not (math.isfinite(flo) and math.isfinite(fhi)) or flo * fhi > 0.0:
        for hi_try in (7.5, 10.0):
            fhi_try = f(hi_try)
            if math.isfinite(fhi_try) and flo * fhi_try <= 0.0:
                hi, fhi = hi_try, fhi_try
                break
        else:
            return (math.nan, "failed", "no_bracket")

    # Bisection fallback
    bres = bisection_basic(f, lo, hi, tol=tol, max_iter=max_iter)
    if bres.get("converged", False):
        return (float(bres["root"]), "bisection", "ok")

    return (math.nan, "failed", bres.get("reason", "no_convergence"))


__all__ = ["implied_vol_bs"]
