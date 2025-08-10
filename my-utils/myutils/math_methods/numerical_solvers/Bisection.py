# myutils/math_methods/numerical_solvers/Bisection.py
import math
from typing import Callable, Dict

def bisection_basic(
    f: Callable[[float], float],
    lo: float,
    hi: float,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> Dict[str, float | int | bool | str]:
    """
    Minimal 1D bisection root finder for solving f(x) = 0 on [lo, hi].

    Requirements
    ------------
    - f(lo) and f(hi) must be finite and have opposite signs
    - lo < hi

    Parameters
    ----------
    f : callable
        Function returning f(x).
    lo, hi : float
        Initial bracketing interval with a sign change.
    tol : float
        Convergence tolerance. Stops when interval width <= tol or |f(mid)| <= tol.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    dict with keys:
        - root (float): final midpoint
        - converged (bool)
        - iterations (int)
        - fval (float): f(root)
        - reason (str): 'ok', 'no_bracket', 'nan_in_eval', 'max_iter', 'bad_interval'

    Notes
    -----
    - This is guaranteed (and very robust) if a valid sign-changing bracket is provided.
    - No logging/printing; handle messages where you call this.
    """
    if not (lo < hi):
        return {"root": lo, "converged": False, "iterations": 0, "fval": float("nan"), "reason": "bad_interval"}

    flo = f(lo)
    fhi = f(hi)

    if not (math.isfinite(flo) and math.isfinite(fhi)):
        return {"root": lo, "converged": False, "iterations": 0, "fval": float("nan"), "reason": "nan_in_eval"}

    # If an endpoint is already (close to) a root:
    if abs(flo) <= tol:
        return {"root": lo, "converged": True, "iterations": 0, "fval": flo, "reason": "ok"}
    if abs(fhi) <= tol:
        return {"root": hi, "converged": True, "iterations": 0, "fval": fhi, "reason": "ok"}

    # Require a sign change across [lo, hi]
    if flo * fhi > 0.0:
        return {"root": lo, "converged": False, "iterations": 0, "fval": float("nan"), "reason": "no_bracket"}

    a, b = lo, hi
    fa, fb = flo, fhi

    for it in range(1, max_iter + 1):
        mid = 0.5 * (a + b)
        fm = f(mid)
        if not math.isfinite(fm):
            return {"root": mid, "converged": False, "iterations": it - 1, "fval": fm, "reason": "nan_in_eval"}

        # Convergence by function value or interval width
        if abs(fm) <= tol or (b - a) <= tol:
            return {"root": mid, "converged": True, "iterations": it, "fval": fm, "reason": "ok"}

        # Keep the half that contains the sign change
        if fa * fm <= 0.0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm

    # Max iterations reached
    mid = 0.5 * (a + b)
    fm = f(mid)
    return {"root": mid, "converged": False, "iterations": max_iter, "fval": fm, "reason": "max_iter"}
