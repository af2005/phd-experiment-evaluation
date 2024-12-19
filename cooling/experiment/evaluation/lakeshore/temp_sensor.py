def systematical_error(temperature: float):
    """
    Temperature readout of Lakeshore Cernox 1070, calibrated temperature sensor
        :return: uncertainty in Kelvin
    """
    if temperature < 10:
        return 4e-3
    if temperature < 20:
        return 8e-3
    if temperature < 30:
        return 9e-3
    if temperature < 50:
        return 12e-3
    if temperature < 100:
        return 16e-3
    if temperature < 300:
        return 45e-3
    return 72e-3
