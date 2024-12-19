import abc
import math
from typing import Union

import scipy.constants as const
from pydantic import PositiveFloat
from uncertainties import ufloat, umath


class Gas(abc.ABC):
    """
    An abstract class for a specific gas type.
    """

    def __init__(
        self,
        name: str,
        mass: Union[float, ufloat],
        temperature: Union[float, ufloat],
        molecular_diameter: Union[float, ufloat],
        pressure: Union[float, ufloat],
    ):
        self.name = name
        self.mass = mass
        self.temperature = temperature
        self.molecular_diameter = molecular_diameter
        self.pressure = pressure

    def mean_velocity(self):
        return math.sqrt(8 * const.k * self.temperature / (self.mass * math.pi))

    @property
    def mean_free_path(self) -> float:
        return (
            const.k
            * self.temperature
            / (math.sqrt(2) * const.pi * self.pressure * self.molecular_diameter**2)
        )

    @property
    def thermal_conductivity(self):
        raise NotImplementedError


class GasCooling:
    def __init__(
        self,
        gas: Gas,
        accommodation_coefficient: float,
        temperature_cold: Union[PositiveFloat, ufloat],
        temperature_hot: Union[PositiveFloat, ufloat],
        area: PositiveFloat,
    ):
        self.gas = gas
        self.alpha = accommodation_coefficient
        self.temperature_cold = temperature_cold
        self.temperature_hot = temperature_hot
        self.area = area

    def molecular_flow_cooling(self):
        return (
            self.alpha
            * umath.sqrt(
                8 * const.k / (const.pi * self.gas.mass * self.temperature_cold)
            )
            * self.gas.pressure
            * self.area
            * (self.temperature_hot - self.temperature_cold)
        )

    def viscous_cooling(self):
        raise NotImplementedError

    def transitional_cooling(self):
        raise NotImplementedError
