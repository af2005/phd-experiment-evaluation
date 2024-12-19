import numpy as np
from scipy.odr import RealData


def sigmuid(p, x):
    a, b, c, d = p
    ex = np.exp((x + c) * b)
    return a + d * ex / (1 + ex)


def polynom_fifth_grade(p, x):
    a, b, c, d, e, f = p
    return a + b * x + c * x**2 + d * x**3 + e * x**4 + f * x**5


def polynom_forth_grade(p, x):
    a, b, c, d, e = p
    return a + b * x + c * x**2 + d * x**3 + e * x**4


def polynom_third_grade(p, x):
    a, b, c, d = p
    return a + b * x + c * x**2 + d * x**3


def polynom_second_grade(p, x):
    a, b, c = p
    return a + b * x + c * x**2


def polynom_second_grade_without_offset(p, x):
    b, c = p
    return b * x + c * x**2


def linear_function(x, m, c):
    return m * x + c


def linear_function_for_ocr(p, x):
    m, c = p
    return m * x + c


def one_over_x(p, x):
    m, c = p
    return m / (x**2) + c


class RealUncertainData(RealData):
    """
    Extends the odr.RealData class to work with uncertainties.ufloat
    """

    def __init__(self, x, y):
        nx = x
        ny = y
        sx = sy = None

        try:
            nx = [p.nominal_value for p in x]
            sx = [p.std_dev for p in x]
        except AttributeError:
            pass
        try:
            ny = [p.nominal_value for p in y]
            sy = [p.std_dev for p in y]
        except AttributeError:
            pass
        super().__init__(x=nx, y=ny, sx=sx, sy=sy)
