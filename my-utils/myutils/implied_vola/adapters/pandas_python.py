# myutils/implied_vola/adapters/pandas_.py
from __future__ import annotations
from typing import Dict, Tuple, Iterable
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from myutils.implied_vola import black_scholes

def _coerce_is_call_vectorized(series: pd.Series) -> np.ndarray:
    def _coerce(val):
        if isinstance(val, bool): return val
        if isinstance(val, (int, float)): return bool(int(val))
        if isinstance(val, str):
            t = val.strip().lower()
            if t in ("c", "call", "calls"): return True
            if t in ("p", "put", "puts"):  return False
        raise ValueError(f"Cannot interpret is_call value: {val!r}")
    return series.apply(_coerce).to_numpy(dtype=bool)

# top-level worker so it’s picklable for ProcessPool
def _solve_one_iv(args: Tuple[float,float,float,float,float,float,bool,float,float,int,Tuple[float,float],bool]):
    price, S, K, T, r, q, is_call, sigma_init, tol, max_iter, bounds, fallback = args
    return black_scholes.bs_price(
        price=price, S=S, K=K, T=T, r=r, q=q, is_call=is_call,
        sigma_init=sigma_init, tol=tol, max_iter=max_iter,
        bounds=bounds, fallback_to_bisection=fallback
    )  # returns (sigma, method, reason)

def compute_iv_on_dataframe(
    df: pd.DataFrame,
    *,
    column_map: Dict[str, str],
    price_preference: str = "mid",      # 'mid' | 'bid' | 'ask'
    dte_unit: str = "days",             # 'days' or 'years'
    rates_in_percent: bool = False,
    sigma_init: float = 0.2,
    tol: float = 1e-6,
    max_iter: int = 100,
    bounds: Tuple[float, float] = (1e-6, 5.0),
    fallback_to_bisection: bool = True,
    out_iv_col: str = "implied_volatility",
    out_method_col: str = "iv_method",
    out_reason_col: str = "iv_reason",
    n_jobs: int = 1,                    # <=—— NEW: set >1 to parallelize solves
) -> pd.DataFrame:
    """
    Semi-vectorized Black–Scholes IV on a DataFrame with optional parallel solves.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    column_map : dict
        Mapping of required inputs to column names:
            - 'S' (spot), 'K' (strike), 'r' (rate), 'q' (dividend), 'is_call'
            - and either 'T' (years) or 'DTE' (days)
            - plus either 'price' OR both 'bid' and 'ask'
    price_preference : {'mid', 'bid', 'ask'}, default 'mid'
        Used only when 'bid' and 'ask' are provided.
    dte_unit : {'days', 'years'}, default 'days'
        Interpret 'T' as years or days; 'DTE' is always days and is converted to years.
    rates_in_percent : bool, default False
        If True, divides r and q by 100.
    sigma_init : float, default 0.2
        Initial guess for each row’s Newton solve.
    tol : float, default 1e-6
        Convergence tolerance for solvers.
    max_iter : int, default 100
        Maximum iterations per solver.
    bounds : (float, float), default (1e-6, 5.0)
        Sigma bounds used by the solver.
    fallback_to_bisection : bool, default True
        Try bisection if Newton fails.
    out_iv_col : str, default 'implied_volatility'
        Name of output IV column.
    out_method_col : str, default 'iv_method'
        Name of output method column.
    out_reason_col : str, default 'iv_reason'
        Name of output reason column.
    n_jobs : int, default 1
        Number of processes for parallel solving (>=2 enables ProcessPoolExecutor).

    Returns
    -------
    pandas.DataFrame
        A copy of `df` with three added columns:
        - out_iv_col (float): implied volatility (NaN if not computed).
        - out_method_col (str): 'newton', 'bisection', 'short_circuit', or 'failed'.
        - out_reason_col (str): 'ok' or a short failure reason.

    Notes
    -----
    - Vectorizes input prep and no-arbitrage bound checks; solves per-row only where needed.
    - Flags rows outside bounds as 'price_out_of_bounds' and skips solving.
    - Marks prices at the lower bound as IV=0 with 'short_circuit'.
    - On Windows or some IDEs, wrap calls with n_jobs>1 under
      `if __name__ == "__main__":` to enable multiprocessing safely.
    """
    # --- validate mapping ---
    required = {"S", "K", "r", "q", "is_call"}
    if not ({"price"} <= column_map.keys() or {"bid", "ask"} <= column_map.keys()):
        raise ValueError("column_map must contain either 'price' or both 'bid' and 'ask'.")
    if not required.issubset(column_map.keys()):
        missing = required - set(column_map.keys())
        raise ValueError(f"column_map is missing required keys: {missing}")
    if not (("T" in column_map) or ("DTE" in column_map)):
        raise ValueError("Provide either 'T' (years) or 'DTE' (days) in column_map.")
    if price_preference not in ("mid", "bid", "ask"):
        raise ValueError("price_preference must be 'mid', 'bid', or 'ask'.")

    out = df.copy()

    # --- vectorized prep ---
    S = out[column_map["S"]].to_numpy(dtype=float)
    K = out[column_map["K"]].to_numpy(dtype=float)
    if "T" in column_map and dte_unit == "years":
        T = out[column_map["T"]].to_numpy(dtype=float)
    elif "T" in column_map and dte_unit == "days":
        T = out[column_map["T"]].to_numpy(dtype=float) / 365.0
    else:
        T = out[column_map["DTE"]].to_numpy(dtype=float) / 365.0

    r = out[column_map["r"]].to_numpy(dtype=float)
    q = out[column_map["q"]].to_numpy(dtype=float)
    if rates_in_percent:
        r = r / 100.0
        q = q / 100.0

    if "price" in column_map:
        price = out[column_map["price"]].to_numpy(dtype=float)
    else:
        bid = out[column_map["bid"]].to_numpy(dtype=float)
        ask = out[column_map["ask"]].to_numpy(dtype=float)
        price = (bid + ask) / 2.0 if price_preference == "mid" else (bid if price_preference == "bid" else ask)

    is_call = _coerce_is_call_vectorized(out[column_map["is_call"]])
    n = len(out)

    # --- vectorized bounds + input checks ---
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)

    lower = np.where(is_call,
                     np.maximum(S * disc_q - K * disc_r, 0.0),
                     np.maximum(K * disc_r - S * disc_q, 0.0))
    upper = np.where(is_call, S * disc_q, K * disc_r)

    eps = 1e-10
    ok_inputs = (S > 0) & (K > 0) & (T > 0)
    within_bounds = (price >= lower - eps) & (price <= upper + eps)
    need_solver = ok_inputs & within_bounds

    # --- preallocate outputs ---
    iv = np.full(n, np.nan, dtype=float)
    method = np.empty(n, dtype=object)
    reason = np.empty(n, dtype=object)

    method[~need_solver] = "failed"
    reason[~ok_inputs] = "bad_inputs"
    bad_bounds_mask = (~within_bounds) & ok_inputs
    reason[bad_bounds_mask] = "price_out_of_bounds"

    # short-circuits
    near_lower = need_solver & (np.abs(price - lower) <= 1e-8)
    iv[near_lower] = 0.0
    method[near_lower] = "short_circuit"
    reason[near_lower] = "at_lower_bound"

    near_upper = need_solver & (np.abs(price - upper) <= 1e-8)
    method[near_upper] = "failed"
    reason[near_upper] = "near_upper_bound"

    # rows to actually solve
    to_solve = need_solver & (~near_lower) & (~near_upper)
    idxs = np.nonzero(to_solve)[0]

    if len(idxs) == 0:
        out[out_iv_col], out[out_method_col], out[out_reason_col] = iv, method, reason
        return out

    # shared parameters
    base_args = (float(sigma_init), float(tol), int(max_iter), (float(bounds[0]), float(bounds[1])), bool(fallback_to_bisection))

    if n_jobs > 1:
        # Build small per-row task tuples (cheap to pickle)
        tasks = [
            (float(price[i]), float(S[i]), float(K[i]), float(T[i]), float(r[i]), float(q[i]), bool(is_call[i]), *base_args)
            for i in idxs
        ]
        workers = min(n_jobs, len(idxs))
        # NOTE: if you call this from a script, put your top-level call under:
        # if __name__ == "__main__": ...   (Windows requirement)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            # chunksize available on Python 3.8+
            results: Iterable[Tuple[float, str, str]] = ex.map(_solve_one_iv, tasks, chunksize=max(1, len(tasks)//(workers*4)))
        for i, (sigma, m, rea) in zip(idxs, results):
            iv[i] = sigma
            method[i] = m
            reason[i] = rea
    else:
        # single-process tight loop
        for i in idxs:
            sigma, m, rea = _solve_one_iv(
                (float(price[i]), float(S[i]), float(K[i]), float(T[i]), float(r[i]), float(q[i]),
                 bool(is_call[i]), *base_args)
            )
            iv[i] = sigma
            method[i] = m
            reason[i] = rea

    out[out_iv_col] = iv
    out[out_method_col] = method
    out[out_reason_col] = reason
    return out
