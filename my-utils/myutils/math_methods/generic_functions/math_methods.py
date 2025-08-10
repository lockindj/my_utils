# myutils/math_methods/generic_functions/math_methods.py
from math import erf, exp, sqrt, pi

# Normal distribution functions (fast scalar approximation)
def norm_cdf(x):
    """
    Standard normal cumulative distribution function.
    Implemented using the error function (erf)
    """
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def norm_pdf(x):
    """
    Standard normal probability density function.
    Direct formula based on exponential function.
    """
    return (1.0 / sqrt(2.0 * pi)) * exp(-0.5 * x * x)