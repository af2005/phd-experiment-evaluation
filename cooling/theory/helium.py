from . import gas
import os
import sys
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate
from uncertainties import ufloat, umath


class Helium(gas.Gas):
    def __init__(self, temperature=None, pressure=None):
        mass = 6.646477 * 10 ** (-27)
        super().__init__(
            "Helium",
            mass=mass,
            temperature=temperature,
            molecular_diameter=ufloat(260e-12, 3e-12),  # 2.2024e-10,  # 2.60e-10,
            pressure=pressure,
        )

    @cached_property
    def _fit_thermal_conductivity(self):
        try:
            data = pd.read_csv(
                os.path.join(
                    sys.path[0],
                    "data/thermal-conductivity-helium.csv",
                )
            ).set_index("temperature")
        except FileNotFoundError:
            data = pd.read_csv(
                os.path.join(sys.path[0], "thermal-conductivity-helium.csv")
            ).set_index("temperature")
        spl = interpolate.splrep(data.index, data["thermal_conductivity"])
        return spl

    def plot_thermal_conductivity(self):
        data = pd.read_csv("thermal-conductivity-helium.csv").set_index("temperature")

        x2 = np.arange(round(data.index[0]), round(data.index[-1]))
        y2 = interpolate.splev(x2, self._fit_thermal_conductivity)
        plt.plot(data.index, data["thermal_conductivity"], "o", x2, y2)
        plt.xlabel("Temperature (K)")
        plt.ylabel("Thermal conductivity (W/(m*K))")

        plt.savefig("thermal-cond-helium.pdf")

    @property
    def thermal_conductivity(self):
        """
        :return: thermal conductivity in W/(m*K)
        """
        derivative = interpolate.splev(
            self.temperature.nominal_value, self._fit_thermal_conductivity, der=1
        )
        nominal_value = interpolate.splev(
            self.temperature.nominal_value, self._fit_thermal_conductivity
        )
        std = derivative * self.temperature.std_dev
        return ufloat(
            nominal_value,
            std,
            tag="PD Calibration helium temperature",
        )

    @property
    def mean_free_path_frost(self) -> float:
        """
        nach Frost 1975
        :return:
        """
        return 0.1 * 2.87e-3 * self.temperature**1.147 / self.pressure

    @property
    def mean_free_path_leybold(self) -> float:
        """
        Mean free path nach Leybold
        nach https://www.leybold.com/en/knowledge/vacuum-fundamentals/fundamental-physics-of-vacuum/outgassing-and-the-mean-free-path
        :return:
        """
        return 18e-3 / self.pressure

    def viscous_flow_cooling(
        self, surface: float, distance: float, temp_hot_surface, temp_cold_surface
    ):
        """
        Returns the viscous flow cooling of gas between two identical, parallel plates.
        :param surface: Surface of a plate in square meters
        :param distance: Distance between the plates in meters
        :param temp_hot_surface: Temperature of the hot plate in Kelvin
        :param temp_cold_surface: Temperature of the cold plate in Kelvin
        :return: Cooling power in Watt
        """
        return (
            self.thermal_conductivity * surface * (temp_hot_surface - temp_cold_surface) / distance
        )

    @staticmethod
    def pressure_correction_transitional_flow(pressure_warm, diameter_tube, temp_warm, temp_cold):
        A = 6.11
        B = 4.26
        C = 0.52
        X = 2 * diameter_tube * pressure_warm / (temp_warm + temp_cold)

        return pressure_warm * (
            1 + (umath.sqrt(temp_cold / temp_warm) - 1) / (A * X**2 + B * X + C * umath.sqrt(X) + 1)
        )
