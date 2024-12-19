import math

import numpy as np
import scipy.constants as const
import scipy.special
import scipy.stats

from .gas import Gas
from .objects import Cavity, Frame, Mirror, Pendulum


class AnalyticalModel:
    """
    Analytical model to calculate the gas cooling power.
    """

    def __init__(
        self,
        gas: Gas,
        mirror: Mirror,
        frame: Frame,
        pendulum: Pendulum,
        cavity: Cavity,
        hit_rate: int,
    ):
        self.gas = gas
        self.mirror = mirror
        self.frame = frame
        self.pendulum = pendulum
        self.cavity = cavity
        self.hit_rate = hit_rate

    @property
    def damping_by_rate(self):
        """
        Mechanical damping rate due to tangential gas damping of mirror.
        Equivalent to expression above, but given as a function of collision rate.
        rate: rate of impinging atoms
        """

        return 2 * self.gas.mass * self.hit_rate

    def damping_by_pressure(self, pressure):
        return (
            pressure
            * self.mirror.effective_surface
            * math.sqrt(
                2 * self.gas.mass / (math.pi * scipy.constants.k * self.gas.temperature)
            )
        )

    def thermal_noise_viscous_damping(self, f):
        """
        Thermal noise displacement PSD of harmonic oscillator for viscous
        damping
        f: frequency
        T: temperature
        m: effective mass
        gam: viscous damping rate
        """
        return (
            4
            * const.k
            * self.gas.temperature
            * self.damping_by_rate
            / (
                self.mirror.mass**2
                * (self.pendulum.omega_0**2 - (2 * const.pi * f) ** 2) ** 2
                + self.damping_by_rate**2 * (2 * np.pi * f) ** 2
            )
        )

    def psd(self, frequencies):
        return (
            np.sqrt(self.thermal_noise_viscous_damping(frequencies))
            / self.cavity.length
        )

    def heat_transfer_molecular_flow(self, pressure):
        """
        Heat transfer in free molecular flow regime (no internal degrees of
        freedom, as COMSOL)
        pressure : incident gas pressure

        """
        temperature_diff = self.frame.temperature - self.mirror.temperature
        return (
            8
            * np.sqrt(const.k / 8 / const.pi / self.gas.mass / self.gas.temperature)
            * pressure
            * self.mirror.surface_of_sides
            * temperature_diff
        )
