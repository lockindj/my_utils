# myutils/math_methods/numerical_solvers/Newton_Raphson.py
import math
from typing import Callable, Optional, Tuple, Dict

def newton_raphson(
    f: Callable[[float], float],
    fprime: Callable[[float], float],
    x0: float,
    tol: float = 1e-8,
    max_iter: int = 50,
    damping: float = 1.0,
    deriv_min: float = 1e-12,
    bounds: Optional[Tuple[float, float]] = None,
) -> Dict[str, float | int | bool | str]:
    """
    Minimal 1D Newton–Raphson root finder for solving f(x) = 0.

    Parameters
    ----------
    f : callable
        Function returning f(x).
    fprime : callable
        Derivative function returning f'(x).
    x0 : float
        Initial guess.
    tol : float
        Convergence tolerance (used for both |step| and |f(x)| checks).
    max_iter : int
        Maximum number of iterations.
    damping : float
        Scale each Newton step by this factor (1.0 = standard Newton).
    deriv_min : float
        If |f'(x)| < deriv_min, we stop with 'zero_derivative'.
    bounds : (lo, hi), optional
        If given, we clamp each iterate into [lo, hi].

    Returns
    -------
    dict with keys:
        - root (float): final x
        - converged (bool)
        - iterations (int)
        - fval (float): f(root)
        - last_step (float)
        - reason (str): 'ok', 'max_iter', 'zero_derivative', 'nan_in_eval', 'no_progress'

    Notes
    -----
    - No logging/printing; handle messages where you call this function.
    """

    if bounds is not None:
        lo, hi = bounds
        if not (lo < hi):
            raise ValueError("bounds must satisfy lo < hi")
        x = min(max(float(x0), lo), hi)
    else:
        x = float(x0)

    for it in range(1, max_iter + 1):
        fx = f(x)
        if not math.isfinite(fx):
            return {"root": x, "converged": False, "iterations": it - 1,
                    "fval": fx, "last_step": 0.0, "reason": "nan_in_eval"}

        # Either the residual is to small
        if abs(fx) <= tol:
            return {"root": x, "converged": True, "iterations": it - 1,
                    "fval": fx, "last_step": 0.0, "reason": "ok"}

        fpx = fprime(x)
        if (not math.isfinite(fpx)) or abs(fpx) < deriv_min:
            return {"root": x, "converged": False, "iterations": it - 1,
                    "fval": fx, "last_step": 0.0, "reason": "zero_derivative"}

        step = -fx / fpx
        step *= damping

        if step == 0.0:
            return {"root": x, "converged": False, "iterations": it - 1,
                    "fval": fx, "last_step": 0.0, "reason": "no_progress"}

        x_new = x + step
        if bounds is not None:
            lo, hi = bounds
            if x_new < lo:
                x_new = lo
            elif x_new > hi:
                x_new = hi

        # Or the step is insignificant
        if abs(x_new - x) <= tol:
            fx_new = f(x_new)
            return {"root": x_new, "converged": True, "iterations": it,
                    "fval": fx_new, "last_step": x_new - x, "reason": "ok"}

        x = x_new

    # newton_raphson did not converge
    fx = f(x)
    return {"root": x, "converged": False, "iterations": max_iter,
            "fval": fx, "last_step": 0.0, "reason": "max_iter"}
