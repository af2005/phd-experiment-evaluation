from uncertainties.unumpy import log10

from . import models


def specific_heat(T: float) -> float:
    """
    Specific heat capacity for teflon between 4 and 300 Kelvin.
    Data from https://trc.nist.gov/cryogenics/materials/Teflon/Teflon_rev.htm
    :param T: temperature
    :return: specific heat capacity
    """
    a = 31.88256
    b = -166.51949
    c = 352.01879
    d = -393.44232
    e = 259.98072
    f = -104.61429
    g = 24.99276
    h = -3.20792
    i = 0.16503
    return models.log_polynome(T, (a, b, c, d, e, f, g, h, i))


def heat_conductivity(T: float) -> float:
    # src https://trc.nist.gov/cryogenics/materials/Teflon/Teflon_rev.htm
    a = 2.7380
    b = -30.677
    c = 89.430
    d = -136.99
    e = 124.69
    f = -69.556
    g = 23.320
    h = -4.3135
    i = 0.33829
    nominal_value = 10 ** (
        a
        + b * log10(T)
        + c * (log10(T)) ** 2
        + d * (log10(T)) ** 3
        + e * (log10(T)) ** 4
        + f * (log10(T)) ** 5
        + g * (log10(T)) ** 6
        + h * (log10(T)) ** 7
        + i * (log10(T)) ** 8
    )
    return nominal_value
