from uncertainties.unumpy import log10


def log_polynome(temperature, parameters):
    exponent = sum(
        [
            parameter * log10(temperature) ** index
            for index, parameter in enumerate(parameters)
        ]
    )
    return 10**exponent
