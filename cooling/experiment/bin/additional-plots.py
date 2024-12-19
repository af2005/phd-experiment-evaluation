import pandas as pd
from uncertainties import ufloat, umath

from cooling.experiment.evaluation.calibration import HighPressureCalibration
from cooling.theory import gas, objects
from cooling.experiment.evaluation.lakeshore.corrector import (
    plot_test_mass_sensor_characteristics,
    testmass_sensor,
)
import helpers

helpers.figures.initialize()

plot_test_mass_sensor_characteristics()


tm = objects.TestMass(temperature=ufloat(7.433346902194525, 0.0008770416849788866))
frame = objects.ThermalObject(temperature=ufloat(7.402210960829073, 0.0011485584713561746))
he = gas.Helium(temperature=(tm.temperature + frame.temperature) / 2)
print(
    f"visc at base={1000*he.viscous_flow_cooling(surface=tm.surface_total, distance=tm.distance_to_frame, temp_hot_surface=tm.temperature, temp_cold_surface=frame.temperature)}"
)
print(
    f"visc at 1 mK={1e3*he.viscous_flow_cooling(surface=tm.surface_total, distance=tm.distance_to_frame, temp_hot_surface=ufloat(7.026,4e-3), temp_cold_surface=7)} mW"
)

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

# print(ScrewHolder.heating_power())

print(10 ** testmass_sensor.controller_temp_to_logresistence(8))
high_cali = HighPressureCalibration()

print(high_cali.laser_power(voltage=-0.338))


high_cali.plot_discommoding_cooling()
high_cali.plot_discommoding_cooling_div_screw_conduction()
high_cali.plot_pd_calibration()
high_cali.plot_lO_tm_temperature_correction()

print(f"{high_cali.laser_power(voltage=0)=}")
print(
    f"High Pressure Cali: Discommoding at TM 10K and frame 8 "
    f"{high_cali.discommoding_cooling_power_by_delta_temperature(frame_temperature=8, tm_temperature=10)}"
)

print("Base heating at L0?")
print(
    high_cali.discommoding_cooling_power_by_delta_temperature(
        frame_temperature=8, tm_temperature=8.598
    )
    * 1000
)


# Secondary Temp sensor corrector
# sc = SecondaryCorrector()
# print(sc.correct_temperature(incorrect_temperature=8.175))
# tm_temps = np.arange(8, 44)
# fig, ax = plt.subplots()
# ax.plot(
#     tm_temps,
#     tm_temps - sc.correct_temperature(tm_temps, phase="cooling"),
#     label="cooling",
# )
# ax.plot(
#     tm_temps,
#     tm_temps - sc.correct_temperature(tm_temps, phase="heating"),
#     label="heating",
# )
# ax.grid()
# # plt.xlim(10.8,11.2)
# # plt.ylim(0.9,0.95)
# plt.legend()
# ax.set_xlabel("X166120 (K)")
# ax.set_ylabel("X166120 - mean(X166410, X166521) (K)")
#
# plt.savefig("figures/secondary-tm-temp-correction.pdf")

## Area between test mass and sensor
print(f"{tm.radius=}")

max_distance = tm.radius - umath.sqrt(tm.radius**2 - (1.905e-3 / 2) ** 2)
average_distance = 0.7 * max_distance
print(f"{average_distance*1e6}um")
print(
    f"Radiation power at 35 K "
    f"{objects.TestMass(temperature=ufloat(35,0)).radiation_power*1000}mW"
)
