import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from package import gas

import helpers
from matplotlib.colors import LinearSegmentedColormap

helpers.figures.initialize()
he = gas.Helium(temperature=18, pressure=5e-4 * 100)


print(f"{1000*he.mean_free_path} mm")
print(f"Knudsen number: {1000*he.mean_free_path/2}")


xs = np.linspace(5, 100, num=50)
plt.plot(
    xs,
    [
        gas.Helium(temperature=x, pressure=1e-4 * 100).mean_free_path * 1000 / 2
        for x in xs
    ],
    label="$10^{-4}$ mbar",
)

plt.plot(
    xs,
    [
        gas.Helium(temperature=x, pressure=1e-3 * 100).mean_free_path * 1000 / 2
        for x in xs
    ],
    label="$10^{-3}$ mbar",
)
plt.plot(
    xs,
    [
        gas.Helium(temperature=x, pressure=1e-2 * 100).mean_free_path * 1000 / 2
        for x in xs
    ],
    label="$10^{-2}$ mbar",
)
plt.plot(
    xs,
    [
        gas.Helium(temperature=x, pressure=1e-1 * 100).mean_free_path * 1000 / 2
        for x in xs
    ],
    label="$10^{-1}$ mbar",
)
# plt.axhspan(1, 10, facecolor="0.2", alpha=0.1)

N = 50
from_x = np.logspace(-1, 1, N)
for i in range(N - 1):
    print(from_x[i], from_x[i + 1])
    c = i / N / 5 + 0.8
    plt.axhspan(
        from_x[i],
        from_x[i + 1],
        color=(c, c, c),
    )

plt.axhspan(0, 0.1, facecolor=(0.8, 0.8, 0.8))

plt.yscale("log")
plt.text(50, 1, "Transitional Flow", ha="center", va="center", style="italic")
plt.text(50, 75, "Free Molecular Flow", ha="center", va="center", style="italic")
plt.text(50, 3.5e-2, "Viscous Flow", ha="center", va="center", style="italic")
plt.xlabel("Temperature (K)")
plt.ylabel("Knudsen number for helium in 2mm space")
plt.xlim(5, 95)
plt.ylim(1e-2, 5e2)
# plt.xticks([10, 20, 30, 40, 50, 60, 70, 80, 90])
plt.gca().xaxis.grid(True)
plt.gcf().set_size_inches(*helpers.figures.set_size())
plt.legend()
plt.savefig("knudsen.pgf", backend="pgf")
plt.savefig("knudsen.pdf", backend="pgf")
