from typing import Iterable, Union

import models
from uncertainties import ufloat


def specific_heat(temperature) -> Union[Iterable[ufloat], ufloat]:
    """
    Specific heat capacity for Polyimide between 4 and 300 Kelvin.
    Data from https://trc.nist.gov/cryogenics/materials/Polyimide%20Kapton/PolyimideKapton_rev.htm
    :param temperature: temperature
    :return: specific heat capacity
    """
    # src
    if 4 > temperature or temperature > 300:
        raise ValueError("specific heat only known between 4 and 300 Kelvin")

    a = -1.3684
    b = 0.65892
    c = 2.8719
    d = 0.42651
    e = -3.0088
    f = 1.9558
    g = -0.51998
    h = 0.051574
    i = 0
    specific_heat_value = models.log_polynome(temperature, (a, b, c, d, e, f, g, h, i))
    return ufloat(specific_heat_value, 0.03 * specific_heat_value, tag="Specific heat Polyimide")


def heat_conductivity(temperature) -> Union[Iterable[ufloat], ufloat]:
    """
    Heat conductivity for Polyimide between 4 and 300 Kelvin.
    :param temperature:
    :return: Heat conductivity
    """
    # src https://trc.nist.gov/cryogenics/materials/Polyimide%20Kapton/PolyimideKapton_rev.htm
    if 4 > temperature or temperature > 300:
        raise ValueError("specific heat only known between 4 and 300 Kelvin")

    a = 5.73101
    b = -39.5199
    c = 79.9313
    d = -83.8572
    e = 50.9157
    f = -17.9835
    g = 3.42413
    h = -0.27133
    i = 0

    conductivity_value = models.log_polynome(temperature, (a, b, c, d, e, f, g, h, i))

    return ufloat(conductivity_value, 0.02 * conductivity_value, tag="Heat conductivity Polyimide")


if __name__ == "__main__":
    print(specific_heat(7.4))
