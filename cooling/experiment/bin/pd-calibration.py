import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from cooling.experiment.evaluation import baserow, calibration

TEMP_FIELD = "Temperature_Testmass_(K)"
PD_FIELD = "Voltage_Photodiode_(V)"
MEASUREMENT_NAME = "live"

COOLDOWN_DATA = [
    # ("2023-01-17T09:12:00Z", "2023-01-17T09:50:00Z"),
    # ("2023-01-17T11:11:00Z", "2023-01-17T12:04:00Z"),
    # ("2023-01-11T11:30:00Z", "2023-01-13T03:10:00Z"),
    (
        "2023-03-16T18:00:00Z",
        "2023-03-17T19:30:00Z",
    )
]
base_connector = baserow.Connector()
PD_CALIBRATIONS = base_connector.discommoding_cooling_data()


def f(B, x):
    """Linear function y = m*x + b"""
    # B is a vector of the parameters.
    # x is an array of the current x values.
    # x is in the same format as the x passed to Data or RealData.
    #
    # Return an array in the same format as y passed to Data or RealData.
    return B[0] * x + B[1]


if __name__ == "__main__":
    cali = calibration.HighPressureCalibration()

    popt, data = cali._fit_photo_voltage_to_discommoding_cooling_power
    fig, ax = plt.subplots()
    data["start"] = pd.to_datetime(data["start"])
    first_run_data = data[data["start"] <= "2023-03-17T00:00"]
    second_run_data = data[data["start"] > "2023-03-17T00:00"]
    ax.errorbar(
        y=[power * 1000 for power in first_run_data["laser_power"]],
        x=[photo_voltage * 1000 for photo_voltage in first_run_data.index],
        yerr=[power * 1000 for power in first_run_data["laser_power_err"]],
        fmt=",",
        zorder=1,
    )
    ax.errorbar(
        y=[power * 1000 for power in second_run_data["laser_power"]],
        x=[photo_voltage * 1000 for photo_voltage in second_run_data.index],
        yerr=[power * 1000 for power in second_run_data["laser_power_err"]],
        fmt=",",
        zorder=1,
    )

    xs = np.linspace(-1000, 0, num=200)
    ys = cali.linear_function(xs * 1e-3, *popt)
    ax.set_xlabel("Photodiode Voltage (mV)")
    ax.set_ylabel("$P_\mathrm{abso. light} $(mW)")

    ax.plot(xs, ys * 1000)
    plt.savefig("figures/pd-calibration.pdf")

    # ======================================

    temperatures = data["temperatures"]
    photo_voltages = data.index
    photo_voltages_err = data["photo_voltage_err"]
    laser_powers = data["laser_power"]
    laser_powers_err = data["laser_power_err"]
    # ======================================
    # Plot temp vs voltage
    # ======================================

    fig1, ax1 = plt.subplots()
    ax1.errorbar(
        x=data["temperatures"],
        y=[photo_voltage * 1000 for photo_voltage in photo_voltages],
        yerr=[photo_voltage * 1000 for photo_voltage in photo_voltages_err],
        fmt="o",
    )
    ax1.set_xlabel("Equilibrated TM temperature with gas (K)")
    ax1.set_ylabel("Photodiode Voltage (mV)")
    plt.tight_layout()
    plt.grid(visible=True, which="both", axis="both")

    # Fit
    # linear = odr.Model(f)
    # fit_data = odr.RealData(
    #     x=temperatures[:13], y=photo_voltages[:13]
    # )  # , sx=laser_powers_err, sy=photo_voltages_err)
    # fit_model = odr.ODR(fit_data, linear, beta0=[-0.01, -0.0095])
    # fit_result = fit_model.run()
    #
    # print(fit_result.beta)
    # # Plot fit
    # xs = np.arange(7, 100, step=0.5)
    # ys = f(fit_result.beta, xs)
    # ax1.plot(xs, [y * 1000 for y in ys])
    # plt.grid(visible=True, which="both", axis="both")

    fig1.savefig("figures/temp_vs_pd.pdf")

    # ======================================
    # Plot power vs voltage
    # ======================================

    fig2, ax2 = plt.subplots()
    ax2.errorbar(
        x=[power * 1000 for power in laser_powers],
        y=[photo_voltage * 1000 for photo_voltage in photo_voltages],
        xerr=[err * 1000 for err in laser_powers_err],
        yerr=[photo_voltage * 1000 for photo_voltage in photo_voltages_err],
        fmt="o",
    )
    interpolated_f = interp1d(
        x=laser_powers, y=photo_voltages, kind="slinear", fill_value="extrapolate"
    )
    xs = np.linspace(0.003, 0.094, num=100)
    ys = interpolated_f(xs)
    ax2.plot([x * 1000 for x in xs], [y * 1000 for y in ys])
    ax2.set_xlabel("Discommoding cooling (mW)")
    ax2.set_ylabel("Photodiode Voltage (mV)")
    plt.tight_layout()
    plt.grid(visible=True, which="both", axis="both")

    fig2.savefig("figures/power_vs_pd.pdf")

    # ======================================
    # Plot power vs temperature
    # ======================================

    fig3, ax3 = plt.subplots()
    ax3.errorbar(
        x=[power * 1000 for power in laser_powers],
        y=temperatures,
        xerr=[err * 1000 for err in laser_powers_err],
        fmt="o",
    )
    ax3.set_xlabel("Discommoding cooling (mW)")
    ax3.set_ylabel("Test mass temperature (K)")
    plt.tight_layout()
    plt.grid(visible=True, which="both", axis="both")

    fig3.savefig("figures/power_vs_temp.pdf")

    # ======================================
    # Interpolated discommoding cooling power
    # ======================================
    fig4, ax4 = plt.subplots()
    xs = np.linspace(10, 140, num=100)
    ys = [cali.discommoding_cooling_power_by_temperature(x).nominal_value for x in xs]
    ax4.errorbar(
        x=xs,
        y=ys,
        fmt=".",
    )
    ax4.set_ylabel("Discommoding cooling (mW)")
    ax4.set_xlabel("Test mass temperature (K)")
    plt.tight_layout()
    plt.grid(visible=True, which="both", axis="both")

    fig4.savefig("figures/interpolated_power_by_temp.pdf")
