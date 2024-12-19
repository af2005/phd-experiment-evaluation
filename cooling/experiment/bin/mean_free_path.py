import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat

from cooling.theory import Helium

he_warm = Helium(temperature=293, pressure=1e-1 * 100)
diameter_capillary = 0.5715e-3
pressure_capillary = he_warm.pressure_correction_transitional_flow(
    pressure_warm=he_warm.pressure,
    diameter_tube=diameter_capillary,
    temp_warm=he_warm.temperature,
    temp_cold=ufloat(8, 2),
)
he_capillary = Helium(temperature=ufloat(8, 2), pressure=pressure_capillary)

print(
    f"Helium inside capillary: \n"
    f"Mean free path{1000*he_capillary.mean_free_path} mm \n"
    f"Pressure: {he_capillary.pressure/100} mbar"
)
print(f"Knudsen number: {he_capillary.mean_free_path/diameter_capillary}")


xs = np.linspace(5, 100, num=50)
plt.plot(
    xs,
    [Helium(temperature=x, pressure=5e-5 * 100).mean_free_path * 1000 / 2 for x in xs],
    label="5.0e-5 mbar",
)
plt.plot(
    xs,
    [Helium(temperature=x, pressure=1e-4 * 100).mean_free_path * 1000 / 2 for x in xs],
    label="1.0e-4 mbar",
)
plt.plot(
    xs,
    [
        Helium(temperature=x, pressure=2.5e-4 * 100).mean_free_path * 1000 / 2
        for x in xs
    ],
    label="2.5e-4 mbar",
)
plt.plot(
    xs,
    [Helium(temperature=x, pressure=5e-4 * 100).mean_free_path * 1000 / 2 for x in xs],
    label="5.0e-4 mbar",
)
plt.plot(
    xs,
    [Helium(temperature=x, pressure=10e-4 * 100).mean_free_path * 1000 / 2 for x in xs],
    label="1.0e-3 mbar",
)
plt.axhspan(0, 10, facecolor="0.2", alpha=0.2)
plt.yscale("log")
plt.text(50, 5, "Transitional Flow", ha="center", va="center", style="italic")
plt.text(50, 75, "Molecular Flow", ha="center", va="center", style="italic")
plt.xlabel("Temperature (K)")
plt.ylabel("Knudsen number for Helium in 2mm space")
plt.xlim(5, 100)
plt.ylim(1, 1e3)
plt.xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.gca().xaxis.grid(True)
plt.legend()
plt.savefig("figures/knudsen.pdf")
