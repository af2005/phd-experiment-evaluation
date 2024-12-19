"""
This file can be used for manual calculation of the accommodation coefficient
to verify the acc-measurement.py calculations
"""

import scipy.constants as const

from cooling.theory import CylindricalSiliconMirror, Helium

tm = CylindricalSiliconMirror(
    temperature=18.92, diameter=0.0254, mass=0.118, accommodation=1, thickness=0.099
)

frame_temp = 7.299
helium = Helium(temperature=frame_temp, pressure=2e-4 * 100)

gas_pressure_inside_chamber = helium.pressure * (helium.temperature / 293) ** 0.5

absorbed_laser = 27e-3
discommonding_cooling = 7e-3

# print(f"{gas_pressure_inside_chamber/100*10000}e-4 mbar")

factor = ((const.pi * helium.mass * frame_temp) / (8 * const.k)) ** 0.5

print(f"{factor=}")
print(f"{(absorbed_laser-discommonding_cooling)=}")
print(f"{gas_pressure_inside_chamber=}")
print(f"{tm.surface_total=}")
print(f"{tm.temperature=}")
print(f"{frame_temp=}")

print(
    factor
    * (absorbed_laser - discommonding_cooling)
    / (gas_pressure_inside_chamber * tm.surface_total * (tm.temperature - frame_temp))
)
"""
factor=0.03714663742976328+/-1.3407758341508809e-06
fmf_cooling=0.019513565572709696+/-0.002942756520439645
self.gas.pressure=0.0040299404394609235+/-0.00042431386848906083
self.mirror.surface_area=0.008833467391510707+/-0.0001978998908345196
self.mirror.temperature=< mirror temp = 18.968178440969186+/-0.014079794668335251 >
self.frame.temperature=< frame temp = 7.299126737043883+/-0.000526911366227281 >
"""
