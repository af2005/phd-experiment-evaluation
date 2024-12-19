import pandas as pd
from scipy.interpolate import splev, splrep
import numpy as np
import matplotlib.pyplot as plt
import helpers.figures
import os

data = pd.read_csv(f"{os.getcwd()}/experiment/pfeiffer_uncertainties.csv")


def _fit_pfeiffer_error(data):
    return splrep(x=data["pressure (hPa)"], y=data["uncertainty"])


def pfeiffer_relative_systematic_error(pressure):
    pressure = pressure / 100  # for Pa to hPa/mbar
    # fit_x = np.logspace(-5, -1)
    # fit_y = splev(fit_x, _fit_pfeiffer_error(data))
    # plt.plot(fit_x, fit_y, label="fit")
    # plt.plot(data["pressure (hPa)"], data["uncertainty"])
    # plt.xscale("log")
    # plt.legend()
    # plt.savefig("pfeiffer_error.pdf")
    return 2 * splev(pressure, _fit_pfeiffer_error(data), ext=2) / 100


def plot_pfeiffer_relative_error():
    helpers.figures.initialize()
    fit_x = np.logspace(-5, -1)
    fit_y = splev(fit_x, _fit_pfeiffer_error(data))
    plt.plot(fit_x, fit_y, label="fit")
    # plt.plot(data["pressure (hPa)"], data["uncertainty"])
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(0.1, 100)
    plt.xlabel("Pressure (mbar)")
    plt.ylabel("Systematic\nuncertainty (%)")
    plt.gcf().set_size_inches(
        helpers.figures.set_size()[0], helpers.figures.set_size()[1] / 2
    )
    plt.grid(which="major", axis="y", linestyle="--")
    plt.savefig("pfeiffer_error.pgf")


if __name__ == "__main__":
    # print(pfeiffer_relative_systematic_error(1e-1 * 100))
    plot_pfeiffer_relative_error()
