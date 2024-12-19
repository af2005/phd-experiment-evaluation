def thermal_expansion(temperature):
    """
    From https://nvlpubs.nist.gov/nistpubs/Legacy/MONO/nistmonograph177.pdf

    Copper expands and contracts its length L, dependent on temperature.
    Baseline is room temperature T0 = 293 K

    :param temperature in Kelvin
    :return: (L(T0) - L(temperature)) / (L(T0) * (T0 - T) )
    """
    if 4 <= temperature <= 300:
        return (11.32 + 3.933e-2 * temperature - 7.306e-5 * temperature**2) * 1e-6
    raise ArithmeticError(f"Cannot calculate thermal expansion of copper for {temperature} K")
