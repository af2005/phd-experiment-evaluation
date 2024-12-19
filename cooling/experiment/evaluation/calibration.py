from abc import ABC
from functools import cached_property
from typing import Iterable, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.odr import ODR, Model
from uncertainties import ufloat, umath

from cooling.theory.data import materials
from . import baserow, fitting, influx
from cooling.theory import Helium, TestMass, ThermalObject
from .lakeshore import temp_sensor
import helpers
from scipy import constants as const

helpers.figures.initialize()

TM = TestMass(temperature=None)
TM_SENSOR_SYSTEMATIC = 31e-3


def print_err(var: ufloat):
    print("########")
    for var, er in var.error_components().items():
        print(var.tag, er)
    print("########")


def get_influx_field_ufloats(fields: list, start, stop):
    return (
        ufloat(data.mean(), data.std())
        for data in [
            influx.InfluxDataContainer(
                start=start,
                stop=stop,
                field=field,
            ).data[field]
            for field in fields
        ]
    )


def get_influx_field_arrays(fields: list, start, stop):
    return (
        data
        for data in [
            influx.InfluxDataContainer(
                start=start,
                stop=stop,
                field=field,
            ).data[field]
            for field in fields
        ]
    )


class Calibration(influx.InfluxDataContainer):
    def __init__(self):
        # self.secondary_corrector = corrector.SecondaryCorrector()
        self.recursion_depth = 0
        self.recursion_depth_max = 3
        super().__init__()

    def relevant_data(self, alpha_measurement_data: pd.DataFrame):
        raise NotImplementedError

    @staticmethod
    def discommoding_cooling_power(
        measurement_data: pd.DataFrame,
    ) -> Union[float, ufloat]:
        raise NotImplementedError

    def discommoding_cooling_power_by_temperature(self, temperature) -> float:
        raise NotImplementedError

    def laser_power(self, voltage: float):
        raise NotImplementedError

    def simulate_cooldown(
        self, starting_temperature: float, seconds: int, step_size: int = 1
    ) -> Iterable:
        """
        To check the calculated heating_powers. This function reverses the process.
        It starts at starting_temperature
        and calculates the test mass temperature over time until stop_time
        :param step_size:
        :param starting_temperature:
        :param seconds: Number of seconds to simulate
        :return: pd.DataFrame with time as index and temperatures
        """
        # starting_temperature = self.data["_value"][3]
        time_stamps = np.arange(0, seconds, step=step_size)
        si = materials.silicon.Silicon()

        temperatures = [starting_temperature]
        specific_heat_capacities = [si.specific_heat(starting_temperature)]

        temperature = starting_temperature
        for _ in time_stamps[1:]:
            heat = step_size * self.discommoding_cooling_power_by_temperature(
                temperature
            )
            temperature -= heat / (TM.mass * si.specific_heat(temperature))
            temperatures.append(temperature)
            specific_heat_capacities.append(si.specific_heat(temperature))

        return pd.DataFrame(
            index=time_stamps,
            data={
                "temperatures": temperatures,
                "specific_heat_capacities": specific_heat_capacities,
            },
        )


class HighPressureCalibration(Calibration, ABC):
    """
    This class handles an alternative calculation to relate the photo voltage seen on the photodiode
    to actual test mass heating.
    The laser power is linear proportional to the photo diode voltage.
        The following effects are influencing the test mass' temperatures:
        1. laser power
        2. gas (different in free molecular flow and viscous flow)
        3. discommoding cooling / heating
            3a. Conduction via teflon screws
            3b. Radiating power from test mass (cooling effect)
            3c. Radiating power from steel (heating effect), due to higher steel temperature

        We calibrate the proportionality factor with two measurements runs.
        One at high pressure, where viscous cooling dominates
        One at low pressure, where discommoding cooling dominates.


    In order to do so, we heat up the TM, switch off the laser
    and analyse the cool-down process.
    Optimally, we only have two heat dissipation channels:
    Radiation and conduction through the teflon screws.
    The gas pressure has to be low enough to result in insignificant gas cooling.
    """

    def __init__(self):
        super().__init__()
        self.baserow_connection = baserow.Connector()
        self.calibration_metadata = (
            self.baserow_connection.pd_calibration_at_high_pressure()
        )

    @cached_property
    def _pd_calibration_data(self) -> Tuple[list[ufloat], list[ufloat], list[ufloat]]:
        """
        Checked manually
        """
        voltages = [ufloat(-200e-6, 800e-6)]
        laser_powers = [ufloat(0, 1e-8)]
        delta_temperatures = [ufloat(0, TM_SENSOR_SYSTEMATIC)]
        for calibration_metadate in self.calibration_metadata:
            testmass_temperature_data = influx.InfluxDataContainer(
                start=calibration_metadate["Start"],
                stop=calibration_metadate["Stop"],
                field="Temperature_Testmass_(K)",
            ).data["Temperature_Testmass_(K)"]
            frame_temperature_data = influx.InfluxDataContainer(
                start=calibration_metadate["Start"],
                stop=calibration_metadate["Stop"],
                field="Temperature_Holver_(K)",
            ).data["Temperature_Holver_(K)"]

            pd_voltage_data = influx.InfluxDataContainer(
                start=calibration_metadate["Start"],
                stop=calibration_metadate["Stop"],
                field="Voltage_Photodiode_(V)",
            ).data["Voltage_Photodiode_(V)"]

            tm_temperature = testmass_temperature_data.mean() - ufloat(
                31e-3, TM_SENSOR_SYSTEMATIC, tag="tm temp correction viscous gas"
            )
            mirror = TestMass(
                temperature=ufloat(
                    tm_temperature.nominal_value,
                    umath.sqrt(
                        testmass_temperature_data.std() ** 2
                        + temp_sensor.systematical_error(tm_temperature) ** 2
                    ),
                    tag="mirror temperature (calibration)",
                )
            )
            frame = ThermalObject(
                temperature=ufloat(
                    frame_temperature_data.mean(),
                    umath.sqrt(
                        frame_temperature_data.std() ** 2
                        + temp_sensor.systematical_error(frame_temperature_data.mean())
                        ** 2
                    ),
                    tag="Frame temperature (calibration)",
                )
            )

            pd_voltage = ufloat(
                pd_voltage_data.mean(),
                pd_voltage_data.std(),
                tag="Photodiode (calibration)",
            )
            proportion_of_frame_temp_to_gas_temp = ufloat(
                0.5, 0.01, tag="pd calibration gas temp"
            )
            helium = Helium(
                temperature=(
                    (1 - proportion_of_frame_temp_to_gas_temp) * mirror.temperature
                    + proportion_of_frame_temp_to_gas_temp * frame.temperature
                ),
                pressure=1.3e3 * 1e-4 * 100,
            )
            viscous_cooling = helium.viscous_flow_cooling(
                surface=mirror.surface_total,
                temp_hot_surface=mirror.temperature,
                temp_cold_surface=frame.temperature,
                distance=mirror.distance_to_frame,
            )
            # print(f"Calibration Knudsen:{helium.mean_free_path / mirror.distance_to_frame}")
            # At base temperature, we see a temp difference between frame and TM,
            # only being explained by a discommoding heating radiating from the steel

            voltages.append(pd_voltage)
            laser_powers.append(viscous_cooling)
            delta_temperatures.append(mirror.temperature - frame.temperature)
            # laser_powers.append(viscous_cooling)
            # delta_temperatures.append(mirror.temperature - frame.temperature - ufloat(26e-3,4e-3))
        return voltages, laser_powers, delta_temperatures

    @staticmethod
    def screw_temperature(frame_temperature, tm_temperature):
        """
        The one end of the screw is at frame temp, the other, pointed end has tm_temp
        :param frame_temperature:
        :param tm_temperature:
        :return:
        """
        factor = 0.81
        return (1 - factor) * tm_temperature + factor * frame_temperature

    def plot_discommoding_cooling_div_screw_conduction(self):
        fit_params, model, df = self._fit_discommoding_cooling_power_by_deltatemperature
        x = np.linspace(0, 20)
        ys = model.fcn(fit_params, x)

        plt.errorbar(x=x, y=[y.nominal_value for y in ys], yerr=[y.std_dev for y in ys])
        plt.grid()
        # .xlim(0, 30)
        plt.errorbar(
            x=[u.nominal_value for u in df["delta T ufloat"]],
            y=[p.nominal_value for p in df["dis_powers_div_conductivity"]],
            xerr=[u.std_dev for u in df["delta T ufloat"]],
            yerr=[p.std_dev for p in df["dis_powers_div_conductivity"]],
            ls="none",
        )
        plt.xlabel("Delta T")

        plt.savefig("figures/high-pressure-discommoding-div-conduction.pdf")
        plt.close()

    def plot_discommoding_cooling(self):
        _, _, df = self._fit_discommoding_cooling_power_by_deltatemperature
        fig1, ax = plt.subplots(figsize=helpers.figures.set_size(1, (1, 1)))
        x = np.linspace(0, 25)
        ys = 1000 * self.discommoding_cooling_power_by_delta_temperature(
            tm_temperature=x + 8, frame_temperature=8
        )
        ax.errorbar(
            x=x, y=[y.nominal_value for y in ys], c="C1", label="Fit (with uncertainty)"
        )
        ax.fill_between(
            x,
            y1=[y.nominal_value - y.std_dev for y in ys],
            y2=[y.nominal_value + y.std_dev for y in ys],
            facecolor="C1",
            alpha=0.2,
            lw=0,
        )

        df["screw_temperature"] = self.screw_temperature(
            tm_temperature=df.index + 8, frame_temperature=8
        )
        df["screw_heat_conductivity"] = materials.teflon.heat_conductivity(
            df["screw_temperature"]
        )
        df["dis_powers"] = (
            df["dis_powers_div_conductivity"] * df["screw_heat_conductivity"]
        )
        # plt.xlim(2, 25)
        # plt.ylim(2, 4)
        ax.errorbar(
            x=[i.nominal_value for i in df.index],
            xerr=[i.std_dev for i in df.index],
            y=[p.nominal_value for p in 1000 * df["dis_powers"]],
            yerr=[p.std_dev for p in 1000 * df["dis_powers"]],
            marker="",
            linestyle="",
            label="Measured data",
        )
        xlabel = plt.xlabel("$T_\mathrm{TM} - T_\mathrm{frame}$ (K)")
        ylabel = plt.ylabel("$\Delta P_{\mathrm{res}}$ (mW)")

        # theory
        deltaT = np.linspace(0, 25)
        frame_temp = 8
        conduction = (
            7  # screws
            * 0.3e-3**2  # radius squared
            * np.pi
            * materials.teflon.heat_conductivity((deltaT + frame_temp * 2) / 2)
            * deltaT
            / 2e-3  # mm distance
        )
        emissivity_silicon = 1
        radiation = (
            emissivity_silicon
            * TM.surface_total.n
            * const.Stefan_Boltzmann
            * (deltaT + frame_temp) ** 4
        )
        ax.plot(
            deltaT,
            conduction * 1000,
            color="C2",
            linestyle="--",
            label="$P_\mathrm{conduction}$ (theory)",
            lw=1,
        )
        ax.plot(
            deltaT,
            radiation * 1000,
            color="C3",
            linestyle="--",
            label="$P_\mathrm{radiation}$ (theory)",
            lw=1,
        )
        ax.plot(
            deltaT,
            (conduction + radiation) * 1000,
            color="C4",
            label="$P_\mathrm{conduction}+P_\mathrm{radiation}$ (theory)",
            lw=1,
        )
        plt.legend()
        plt.xlim(0, 25)

        fig1.savefig(
            "figures/high-pressure-discommoding.pgf",
            backend="pgf",
            # bbox_extra_artists=[xlabel, ylabel],
            # bbox_inches="tight",
        )
        plt.close()

    @cached_property
    def _fit_L0_tm_temperature_correction(self):
        metadata = self.baserow_connection.low_pressure_tm_temperature_correction()
        tm_temperatures = []
        corrections = []
        for correction in metadata:
            tm_temperature, frame_temperature = get_influx_field_ufloats(
                [
                    "Temperature_Testmass_(K)",
                    "Temperature_Holver_(K)",
                ],
                start=correction["Start"],
                stop=correction["Stop"],
            )
            if abs(tm_temperature.std_dev) > 0.3:
                print(f"Rejecting tm_temp correction at {correction['Start']}")
                continue

            tm_temperatures.append(tm_temperature)
            corrections.append(tm_temperature - frame_temperature)

        real_data = fitting.RealUncertainData(x=tm_temperatures, y=corrections)
        model = Model(fitting.one_over_x)
        fit = ODR(model=model, data=real_data, beta0=[1.0, 1.0]).run()

        df = pd.DataFrame(
            data={
                "corr": [corr.nominal_value for corr in corrections],
                "corr_err": [corr.std_dev for corr in corrections],
            },
            index=tm_temperatures,
        ).sort_index()
        return fit.beta, fit.sd_beta, df

    def l0_tm_temperature_correction(self, false_temperature):
        correction = fitting.one_over_x(
            self._fit_L0_tm_temperature_correction[0], false_temperature
        )
        correction_uncertainty = ufloat(0, 0.5, tag="tm sensor in no gas phase")
        try:
            correction += correction_uncertainty
        except np.core._exceptions._UFuncOutputCastingError:
            pass
        corrected_temperature = false_temperature - correction

        return corrected_temperature

    def plot_lO_tm_temperature_correction(self):
        _, _, df = self._fit_L0_tm_temperature_correction
        ys = df["corr"]
        plt.errorbar(
            x=[i.nominal_value for i in df.index],
            xerr=[i.std_dev for i in df.index],
            yerr=df["corr_err"],
            y=-ys,
            marker="x",
            linestyle="",
        )

        xs = np.linspace(start=7, stop=40, num=50)
        ys = self.l0_tm_temperature_correction(xs) - xs
        plt.plot(xs, ys)
        plt.xlabel("Sensor Testmass temperature (K)")
        plt.ylabel("Correct testmass temperature (K)")
        plt.grid()
        # plt.xlim(22,22.5)
        plt.savefig("figures/L0_tm_temp_correction.pdf")
        # plt.close()

    def discommoding_cooling_power_by_delta_temperature(
        self, frame_temperature, tm_temperature
    ) -> Union[float, np.array]:
        fit_params, model, _ = self._fit_discommoding_cooling_power_by_deltatemperature

        cooling_div_conductivty = model.fcn(
            fit_params, (tm_temperature - frame_temperature)
        )

        heat_conductivity = materials.teflon.heat_conductivity(
            self.screw_temperature(
                frame_temperature=frame_temperature, tm_temperature=tm_temperature
            )
        )
        return cooling_div_conductivty * heat_conductivity

    @cached_property
    def _fit_discommoding_cooling_power_by_deltatemperature(self):
        """

        :return:
        """
        discommoding_calibrations = self.baserow_connection.discommoding_cooling_data()

        discommoding_powers_div_conductivity = [
            ufloat(0, 1e-9, tag="sensor heating base")
        ]
        temperatures = [
            ufloat(0e-3, TM_SENSOR_SYSTEMATIC, tag="tm temp no gas, no laser")
        ]
        starts = ["manual"]
        heating_powers = [
            # self.base_heating
            ufloat(0, 1e-9, tag="sensor heating base")
        ]

        for calibration in discommoding_calibrations:
            pd_voltage, tm_temperature, frame_temperature = get_influx_field_ufloats(
                [
                    "Voltage_Photodiode_(V)",
                    "Temperature_Testmass_(K)",
                    "Temperature_Holver_(K)",
                ],
                start=calibration["Start"],
                stop=calibration["Stop"],
            )

            tm_temperature = ufloat(
                tm_temperature.nominal_value,
                tm_temperature.std_dev,
                tag="tm temp statistical",
            )
            tm_temperature -= ufloat(757e-3, 700e-3, tag="tm temperature correction")
            tm_temperature += ufloat(
                0,
                temp_sensor.systematical_error(tm_temperature.nominal_value),
                tag="tm temp systematically",
            )

            frame_temperature = ufloat(
                frame_temperature.nominal_value,
                umath.sqrt(
                    frame_temperature.std_dev**2
                    + temp_sensor.systematical_error(frame_temperature.nominal_value)
                    ** 2
                ),
                tag="frame temperature sensor",
            )

            if 32 < tm_temperature:
                continue

            if calibration["Name"] == "Base":
                pd_voltage = ufloat(-100e-6, 500e-6)
                print(f"Found base, is now {self.laser_power(pd_voltage)}")
            # print(f"{frame_temperature=}")
            # print(f"{tm_temperature=}")
            discommoding_powers_div_conductivity.append(
                (self.laser_power(pd_voltage))
                / materials.teflon.heat_conductivity(
                    self.screw_temperature(
                        frame_temperature=frame_temperature,
                        tm_temperature=tm_temperature,
                    ).nominal_value
                )
            )

            temperatures.append((tm_temperature - frame_temperature))
            starts.append(calibration["Start"])
            heating_powers.append(self.laser_power(pd_voltage))
        df = (
            pd.DataFrame(
                data={
                    "dis_powers_div_conductivity": discommoding_powers_div_conductivity,
                    "delta T": temperatures,
                    "starts": starts,
                    "delta T ufloat": temperatures,
                }
            )
            .set_index("delta T")
            .sort_index()
        )
        model = Model(fitting.polynom_second_grade_without_offset)
        real_data = fitting.RealUncertainData(
            x=df.index,
            y=df["dis_powers_div_conductivity"].to_numpy(),
        )

        odr = ODR(real_data, model, beta0=[1.0] * 2)

        odr_fit = odr.run()
        df["dis_powers_div_conductivity"] = df["dis_powers_div_conductivity"]
        df["dis_powers_div_cond_fit"] = model.fcn(odr_fit.beta, df.index)
        fit = [
            ufloat(x, sx, tag="unwanted cooling (systematic)")
            for x, sx in zip(odr_fit.beta, odr_fit.sd_beta)
        ]
        return fit, model, df

    @property
    def base_heating_L0(self):
        return 0.0

    @cached_property
    def base_heating(self):
        """
        oldTODO: if this is electrical heating of the sensor itself,
        then it should be dependent on the current applied to it by the controller??
        """
        raise Exception("Do not use.")

        measurement_meta = self.baserow_connection.base_heating_H0()
        base_heatings = []
        for meta in measurement_meta:
            frame = influx.InfluxDataContainer(
                start=meta["Start"],
                stop=meta["Stop"],
                field="Temperature_Holver_(K)",
            ).data["Temperature_Holver_(K)"]
            tm_data = influx.InfluxDataContainer(
                start=meta["Start"],
                stop=meta["Stop"],
                field="Temperature_Testmass_(K)",
            ).data["Temperature_Testmass_(K)"]

            tm = TestMass(
                temperature=ufloat(
                    tm_data.mean(),
                    umath.sqrt(
                        tm_data.std() ** 2
                        + (temp_sensor.systematical_error(tm_data.mean())) ** 2
                    ),
                    tag="Temperature sensor Base Discommoding heating",
                )
            )
            frame_temp = ufloat(
                frame.mean(),
                umath.sqrt(
                    frame.std() ** 2 + temp_sensor.systematical_error(frame.mean()) ** 2
                ),
            )

            base_heatings.append(
                Helium(
                    temperature=(tm.temperature * 0.45 + frame_temp * 0.55)
                ).viscous_flow_cooling(
                    temp_hot_surface=tm.temperature,
                    temp_cold_surface=frame_temp,
                    distance=tm.distance_to_frame,
                    surface=tm.surface_total,
                )
            )
        return sum(base_heatings) / len(base_heatings)

    @cached_property
    def pd_calibration_fit(self) -> Tuple[ufloat, ufloat]:
        voltages, laser_powers, _ = self._pd_calibration_data
        linear_model = Model(fitting.linear_function_for_ocr)
        real_data = fitting.RealUncertainData(
            x=voltages,
            y=laser_powers,
        )
        odr = ODR(real_data, linear_model, beta0=[-2.0, -100e-6])
        fit = odr.run()
        return (
            ufloat(fit.beta[0], fit.sd_beta[0], tag="pd calibration slope"),
            ufloat(fit.beta[1], fit.sd_beta[1], tag="pd calibration intercept"),
        )

    def laser_power(self, voltage: ufloat) -> ufloat:
        """
        :param voltage: Voltage of the photodiode
        :return:
        """
        try:
            if voltage > -500e-6:
                return ufloat(1e-8, 1e-5, tag="pd statistial")
        except ValueError:
            pass
        return fitting.linear_function(voltage, *self.pd_calibration_fit)

    def plot_pd_calibration(self):
        data = self._pd_calibration_data
        x_fit = np.linspace(-2, 0)
        fig1, ax = plt.subplots(figsize=helpers.figures.set_size(1, (1, 1)))

        print("PD Calibration Error Sources")
        for var, error in data[1][0].error_components().items():
            print("{}: {}".format(var.tag, error))

        ax.errorbar(
            x=[point.nominal_value for point in data[0]],
            y=[point.nominal_value * 1000 for point in data[1]],
            xerr=[point.std_dev for point in data[0]],
            yerr=[point.std_dev * 1000 for point in data[1]],
            fmt=".",
            label="Measured data",
        )
        # for count, annotation in enumerate(data[2]):
        #     ax.text(
        #         data[0][count].nominal_value - 0.05,
        #         data[1][count].nominal_value * 1.0,
        #         f"ΔT = {round(annotation.nominal_value,2)} K",
        #         rotation=0,
        #         size=10,
        #         horizontalalignment='right',
        #         verticalalignment='center'
        # )
        ax.plot(
            x_fit,
            [y.nominal_value * 1000 for y in self.laser_power(x_fit)],
            label="Fit",
        )
        plt.legend()
        # plt.xlim(-0.04,0)
        # plt.ylim(0,0.001)
        xlabel = plt.xlabel("Photodiode Voltage (V)")
        ylabel = plt.ylabel("Absorbed laser power (mW)")
        fig1.savefig(
            "figures/high-pressure-pd-calibration.pgf",
            backend="pgf",
            # bbox_extra_artists=[xlabel, ylabel],
            # bbox_inches="tight",
        )
        plt.close()


class MixedCalibration:
    def __init__(self):
        super().__init__()
        self.high_pressure_calibration = HighPressureCalibration()

    def laser_power(self, voltage: float) -> float:
        return self.high_pressure_calibration.laser_power(voltage)


class PowerMeterCalibration(Calibration, ABC):
    @property
    def data(self):
        raise NotImplementedError

    @staticmethod
    def relevant_measurement_data(measurement_data: pd.DataFrame):
        return measurement_data["photo_voltage"]

    def discommoding_cooling_power(
        self, measurement_data: pd.DataFrame
    ) -> Union[float, ufloat]:
        relevant_data = self.relevant_measurement_data(measurement_data)
        return (
            ufloat(abs(relevant_data.mean()), abs(relevant_data.mean()) * 0.1) * 0.192
            - 1.17
        )

    def __str__(self):
        return "Power Meter calibrated (large error)"


if __name__ == "__main__":
    c = HighPressureCalibration()
    print(c.base_heating)
