from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from scipy.interpolate import splev, splrep
from uncertainties import ufloat

from cooling.theory.data.materials import copper, silicon


class ThermalObject:
    def __init__(self, temperature):
        self.temperature = temperature


class ThermalSurface(ThermalObject):
    def __init__(self, surface_total, temperature):
        super().__init__(temperature)
        self.surface_total = surface_total


class ThermalCylinder(ThermalSurface):
    def __init__(self, diameter, thickness, temperature):
        surface_total = 2 * (diameter / 2) ** 2 * np.pi + thickness * np.pi * diameter
        super().__init__(surface_total=surface_total, temperature=temperature)


class Cable:
    def __init__(self, temperature: float, diameter, length):
        self.temperature = temperature
        self.diameter = diameter
        self.length = length

    @property
    def cross_sectional_area(self):
        return const.pi * (self.diameter / 2) ** 2

    @property
    def given_conductivity_data(self):
        raise NotImplementedError

    def _heat_conductivity_spline(self):
        return splrep(
            x=self.given_conductivity_data[0], y=self.given_conductivity_data[1]
        )

    def plot_heat_conductivity(self, start=4, stop=300):
        fitted_temps = np.linspace(start, stop)
        fitted_conductivities = splev(
            fitted_temps, tck=self._heat_conductivity_spline()
        )
        plt.plot(fitted_temps, fitted_conductivities)
        plt.plot(*self.given_conductivity_data)

    @property
    def heat_conductivity(self):
        return splev(self.temperature, tck=self._heat_conductivity_spline())

    def cooling(self, delta_temperature):
        return (
            self.heat_conductivity
            * delta_temperature
            * self.cross_sectional_area
            / self.length
        )


class CablePhosphorBronze(Cable):
    @property
    def given_conductivity_data(self):
        """
        Data from https://www.lakeshore.com/products/categories/specification/temperature-products/cryogenic-accessories/cryogenic-wire
        :return:
        """
        temperatures = [4, 10, 20, 80, 150, 300]  # Kelvins
        conductivities = [1.6, 4.6, 10, 25, 34, 48]  # Watt/(meter K)
        return temperatures, conductivities


class TestMassWire(CablePhosphorBronze):
    def __init__(self, temperature, length):
        super().__init__(temperature=temperature, diameter=0.127e-3, length=length)


class AccommodatingObject:
    def __init__(self, temperature, accommodation, mass):
        self.temperature = temperature
        self.accommodation = accommodation
        self.mass = mass

    @property
    def mass(self):
        return self._mass

    @mass.setter
    def mass(self, mass: float):
        if mass <= 0:
            raise ValueError(f"The mass of {str(self)} must be greater than 0.")
        self._mass = mass


class PointMass:
    def __init__(self, mass, seismic_amplitude, seismic_frequency):
        self.mass = mass
        self.seismic_amplitude = seismic_amplitude
        self.seismic_frequency = seismic_frequency


class Frame(AccommodatingObject):
    def __init__(self, temperature, accommodation, mass):
        super().__init__(
            temperature=temperature, accommodation=accommodation, mass=mass
        )


class Mirror(AccommodatingObject):
    def __init__(self, mass, density, temperature, accommodation, gas_hits_front):
        super().__init__(
            temperature=temperature, accommodation=accommodation, mass=mass
        )
        self.density = density
        self.current_position = 0
        self.velocity = 0
        self.positions = [0]
        self.gas_hits_front = gas_hits_front

    @property
    def surface_of_sides(self) -> float:
        raise NotImplementedError

    @property
    def surface_of_front_and_back(self) -> float:
        raise NotImplementedError

    @property
    def surface_total(self) -> float:
        return self.surface_of_sides + self.surface_of_front_and_back

    @property
    def thickness(self) -> float:
        raise NotImplementedError

    @property
    def effective_surface(self):
        if self.gas_hits_front:
            return self.surface_of_front_and_back + self.surface_of_sides
        return self.surface_of_sides


class SiliconMirror(Mirror):
    def __init__(self, mass, temperature, accommodation, gas_hits_front):
        super().__init__(
            mass=mass,
            density=2330,
            temperature=temperature,
            accommodation=accommodation,
            gas_hits_front=gas_hits_front,
        )


class CylindricalSiliconMirror(SiliconMirror):
    def __init__(
        self,
        mass,
        diameter,
        temperature,
        accommodation,
        thickness,
        gas_hits_front: bool = False,
    ):
        self.diameter = diameter
        self.thickness = thickness
        super().__init__(
            mass=mass,
            temperature=temperature,
            accommodation=accommodation,
            gas_hits_front=gas_hits_front,
        )

    @property
    def thickness_from_density(self) -> float:
        """
        Density is given by the material.
        If we want to vary the mass of the mirror, a simple solution is the change the thickness for a given mass.
        :return:
        """
        return 4 * self.mass / (self.density * np.pi * self.diameter**2)

    @property
    def material(self):
        return silicon.Silicon()

    def length_cold(self, length_warm):
        """
        L(8 K) = L(293 K) - L(293 K) * 215e-6
        https://srd.nist.gov/JPCRD/jpcrd220.pdf
        :return:
        """
        return length_warm - length_warm * 215e-6

    @property
    def surface_of_sides(self) -> float:
        return np.pi * self.diameter * self.thickness

    @property
    def surface_of_front_and_back(self):
        return 2 * np.pi * (self.diameter / 2) ** 2

    def probability_for_hit_on_side_surface(self):
        if self.gas_hits_front:
            return self.surface_of_sides / (
                self.surface_of_sides + self.surface_of_front_and_back
            )
        return 1

    def probability_for_hit_on_front_or_surface(self):
        return 1 - self.probability_for_hit_on_side_surface()

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, value):
        self._thickness = value


class Pendulum:
    def __init__(self, length):
        self.length = length

    @property
    def omega_0(self):
        return np.sqrt(const.g / self.length)

    @property
    def resonance_frequency(self):
        return self.omega_0 / 2 / np.pi


class Cavity:
    def __init__(self, length):
        self.length = length


class TestMass(CylindricalSiliconMirror):
    def __init__(self, temperature):
        super().__init__(
            mass=0.118,
            diameter=self._calc_diameter,
            temperature=temperature,
            accommodation=None,
            thickness=self._calc_thickness,
        )

    @property
    def radius(self):
        return self.diameter / 2

    @property
    def radiation_power(self):
        return (
            0.4 * self.surface_total.n * const.Stefan_Boltzmann * (self.temperature**4)
        )

    @cached_property
    def _calc_diameter(self):
        data = self.length_cold(
            np.array(
                [
                    2.485,
                    2.485,
                    2.485,
                    2.49,
                    2.49,
                    2.485,
                    2.485,
                    2.485,
                    2.49,
                    2.49,
                    2.485,
                ]
            )
            * 1e-2
        )
        return ufloat(data.mean(), data.std(), tag="mirror diameter")

    @cached_property
    def _calc_thickness(self):
        data = np.array([9.98, 9.97, 9.975, 9.97, 9.97, 9.98]) * 1e-2
        return self.length_cold(ufloat(data.mean(), data.std(), tag="mirror thickness"))

    @cached_property
    def distance_to_frame(self):
        return ufloat(0.00224, 0.00007, tag="distance frame tm")


class Holder:
    def __init__(self, tm_temperature, frame_temperature):
        self.tm = TestMass(temperature=tm_temperature)
        self.frame_temperature = frame_temperature

    @property
    def inner_diameter(self):
        data = (
            np.array(
                [2.94, 2.92, 2.96, 2.935, 2.94, 2.97, 2.96, 2.94, 2.935, 2.94, 2.935]
            )
            * 1e-2
        )
        length_warm = ufloat(data.mean(), data.std(), tag="tm holder inner radius")
        length_cold = length_warm - copper.thermal_expansion(
            self.frame_temperature
        ) * length_warm * (LAB_TEMPERATURE - self.frame_temperature)
        return length_cold

    @property
    def inner_radius(self):
        return self.inner_diameter / 2

    @property
    def distance_tm_frame(self):
        return self.inner_radius - (self.tm.diameter / 2)


LAB_TEMPERATURE = ufloat(293, 3, tag="temperature laboratory")

if __name__ == "__main__":
    print(Holder(8, 10).inner_radius)
    print(Holder(8, 10).distance_tm_frame)
    print(Holder(8, 100).distance_tm_frame)
