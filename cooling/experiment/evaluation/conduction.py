import math

from uncertainties import ufloat

from cooling.theory.data.materials.teflon import heat_conductivity


class ScrewHolder:
    def __init__(self, number_of_screws=6):
        self.number_of_screws = number_of_screws
        self.touching_area = 1e-3
        # self.frame_temp = ufloat(6.4, 0.05, tag="discommoding heating")
        self.frame_temp = ufloat(6.4, 0.05, tag="discommoding heating")
        # self.mirror_temp = ufloat(7.16, 0.02, tag="discommoding heating")
        self.mirror_temp = ufloat(7.16, 0.02, tag="discommoding heating")
        self.screw_length = 2e-3

    def heating_power(self):
        return (
            self.number_of_screws
            * heat_conductivity(7)
            * math.pi
            * self.touching_area**2
            * (self.mirror_temp - self.frame_temp)
            / (4 * self.screw_length)
        )


if __name__ == "__main__":
    print(ScrewHolder().heating_power())
