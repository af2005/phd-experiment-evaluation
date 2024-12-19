import numpy as np
import scipy.constants as const
from uncertainties import umath

from cooling.theory.helium import Helium
from cooling.theory import objects


class HighPressureFinder:
    def __init__(self):
        self.frame = objects.ThermalObject(temperature=7.8)
        self.mirror = objects.ThermalObject(temperature=10)
        self.gas_in = Helium(pressure=1, temperature=self.frame.temperature)
        self.gas_out = Helium(pressure=1, temperature=self.mirror.temperature)

    def readout_pressure(self):
        print(
            2 * self.gas_in.pressure * (293 / self.frame.temperature) ** 0.5,
        )

    def knudsen_number(self, accommodation_coefficient: float, distance=2e-3):
        """
        :return:
        """
        # lambda_in = self.gas_in.mean_free_path
        _sum_of_temps = self.frame.temperature + self.mirror.temperature
        product_of_temps = self.frame.temperature * self.mirror.temperature
        sqrt_of_temps = umath.sqrt(self.frame.temperature / self.mirror.temperature)

        v_in_in = v_out2_out2 = umath.sqrt(
            const.k * self.frame.temperature * (8 - np.pi) / self.gas_in.mass
        )
        v_out1_out1 = umath.sqrt(
            const.k * self.mirror.temperature * (8 - np.pi) / self.gas_out.mass
        )
        v_out1_out2 = v_out2_out1 = umath.sqrt(
            (const.k / self.gas_out.mass)
            * (4 * _sum_of_temps - np.pi * umath.sqrt(product_of_temps))
        )

        v_in_out1 = v_out1_in = umath.sqrt(
            (const.k / self.gas_out.mass)
            * (4 * _sum_of_temps + np.pi * umath.sqrt(product_of_temps))
        )
        v_in_out2 = v_out2_in = umath.sqrt(
            const.k * self.frame.temperature * (8 + np.pi) / self.gas_in.mass
        )

        v_in = v_out2 = umath.sqrt(
            9 * np.pi * const.k * self.gas_in.temperature / (8 * self.gas_in.mass)
        )
        v_out1 = umath.sqrt(
            9 * np.pi * const.k * self.gas_out.temperature / (8 * self.gas_out.mass)
        )

        lambda_in = (
            const.k
            * self.frame.temperature
            * v_in
            / (
                np.pi
                * self.gas_in.molecular_diameter**2
                * self.gas_in.pressure
                * (
                    v_in_in
                    + accommodation_coefficient * sqrt_of_temps * v_in_out1
                    + (1 - accommodation_coefficient) * v_in_out2
                )
            )
        )

        lambda_out_1 = (
            const.k
            * self.mirror.temperature
            * v_out1
            / (
                np.pi
                * self.gas_in.molecular_diameter**2
                * self.gas_in.pressure
                * (
                    accommodation_coefficient * sqrt_of_temps * v_out1_out1
                    + v_out1_in
                    + (1 - accommodation_coefficient) * v_out1_out2
                )
            )
        )
        lambda_out_2 = (
            const.k
            * self.frame.temperature
            * v_out2
            / (
                np.pi
                * self.gas_in.molecular_diameter**2
                * self.gas_in.pressure
                * (
                    accommodation_coefficient * sqrt_of_temps * v_out2_out1
                    + v_out2_in
                    + (1 - accommodation_coefficient) * v_out2_out2
                )
            )
        )
        a = (
            lambda_in
            + accommodation_coefficient * sqrt_of_temps * lambda_out_1
            + (1 - accommodation_coefficient) * lambda_out_2
        )
        b = 2 + accommodation_coefficient * (sqrt_of_temps - 1)

        return (a / b) / distance


a = HighPressureFinder()
print(a.knudsen_number(accommodation_coefficient=0.3))
a.readout_pressure()
