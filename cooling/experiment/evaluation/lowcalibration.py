from functools import cached_property
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from uncertainties import ufloat

from . import baserow, influx
from cooling.theory import TestMass, materials
from lakeshore import corrector, temp_sensor

TM = TestMass(temperature=None)


class CooldownDataContainer(influx.InfluxDataContainer):
    def _download_data(self):
        return influx.get_aggregated_data(
            field=self.field,
            start=self.start,
            stop=self.stop,
            measurement=self.measurement,
            aggregation_window="10s",
        )

    def __str__(self):
        return f"Cooldown with data from {self.start[:10]}"


class LowPressureCalibration:
    """
    This class handles the calculation to relate
    equilibrium temperature of the test mass surrounded by gas
    to actual test mass heating.
    This cool down is done without any gas in the system.
    In order to do so, we heat up the TM, switch off the laser and analyse the
    cool down process. Optimally, we only have two heat dissipation channels:
    Radiation and conduction through the teflon screws.
    The gas pressure has to be low enough to result in insignificant gas cooling.
    """

    def __init__(self, temperatures_need_correction=False):
        self.baserow_connection = baserow.Connector()
        self.cooldowns_metadata = self.baserow_connection.cooldowns()
        self.cooldowns = self._download_cooldowns()
        self.temperatures_need_correction = temperatures_need_correction

    def _download_cooldowns(self) -> [CooldownDataContainer]:
        return [
            CooldownDataContainer(
                start=cooldown_meta["Start"],
                stop=cooldown_meta["Stop"],
                field=cooldown_meta["Influx Temperature Field"],
                measurement=cooldown_meta["Influx Measurement Name"],
            )
            for cooldown_meta in self.cooldowns_metadata
            if cooldown_meta["active"]
        ]

    @property
    def data_indexed_by_temperature(self) -> [pd.DataFrame]:
        return [
            cooldown_data.set_index("_value").sort_index()
            for cooldown_data in self.cooldown_data_indexed_by_time
        ]

    @property
    def cooldown_power_indexed_by_temperature(self) -> [pd.Series]:
        """
        :return:Dataframe of real calibration data.
        """
        return [data["heating power (W)"] for data in self.data_indexed_by_temperature]

    @cached_property
    def _interpolate_discommoding_cooling_power(self) -> [Callable]:
        # this also has an error
        return [
            interp1d(
                data.index,
                data.array,
                kind="linear",
            )
            for data in self.cooldown_power_indexed_by_temperature
        ]

    # def discommoding_cooling_power_by_deltatemperature(self, tm_temperature, frame_temperature):
    #     #incorrect
    #     return self.discommoding_cooling_power_by_temperature(tm_temperature - frame_temperature)

    def discommoding_cooling_power_by_temperature(self, temperature: float) -> ufloat:
        """
        :param temperature: The temperature of the test mass
        :return:
        """
        measurement_uncertainty = temp_sensor.systematical_error(temperature)
        try:
            data = [
                interpolation(temperature)
                for interpolation in self._interpolate_discommoding_cooling_power
            ]
            data += [
                interpolation(temperature - measurement_uncertainty)
                for interpolation in self._interpolate_discommoding_cooling_power
            ]
            data += [
                interpolation(temperature + measurement_uncertainty)
                for interpolation in self._interpolate_discommoding_cooling_power
            ]
            # This is just estimated
            return ufloat(np.nanmean(data), np.nanstd(data), tag="discommoding cooling")
        except ValueError:
            return ufloat(0, 0)

    @cached_property
    def _fit_photo_voltage_to_discommoding_cooling_power(self):
        """

        :return: something to be evaluated with scipy.interpolate.splev
        """
        pd_calibrations = self.baserow_connection.discommoding_cooling_data()
        laser_powers = []
        laser_powers_err = []
        photo_voltages = []
        photo_voltages_err = []
        temperatures = []
        starts = []

        for calibration in pd_calibrations:
            temp_field = "Temperature_Testmass_(K)"
            pd_field = "Voltage_Photodiode_(V)"
            temp_data = influx.InfluxDataContainer(
                start=calibration["Start"],
                stop=calibration["Stop"],
                field=temp_field,
                measurement=calibration["Influx Measurement Name"],
            ).data
            pd_data = influx.InfluxDataContainer(
                start=calibration["Start"],
                stop=calibration["Stop"],
                field=pd_field,
                measurement=calibration["Influx Measurement Name"],
            ).data
            temperature = ufloat(temp_data[temp_field].mean(), temp_data[temp_field].std())
            photodiode = ufloat(pd_data[pd_field].mean(), pd_data[pd_field].std())
            print(calibration["Start"])
            laser_power = self.discommoding_cooling_power_by_temperature(temperature.nominal_value)
            starts.append(calibration["Start"])
            temperatures.append(temperature.nominal_value)
            laser_powers.append(laser_power.nominal_value)
            laser_powers_err.append(laser_power.std_dev)
            photo_voltages.append(photodiode.nominal_value)
            photo_voltages_err.append(photodiode.std_dev)
            # print(
            #     f"Messung {calibration['Name']}: @{temperature}K with {photodiode * 1000} mV resulting in {laser_power * 1000} mW "
            # )

        df = (
            pd.DataFrame(
                data={
                    "start": starts,
                    "laser_power": laser_powers,
                    "laser_power_err": laser_powers_err,
                    "photo_voltage": photo_voltages,
                    "photo_voltage_err": photo_voltages_err,
                    "temperatures": temperatures,
                }
            )
            .set_index("photo_voltage")
            .sort_index()
        )

        popt, _ = curve_fit(
            self.linear_function,
            ydata=df["laser_power"],
            xdata=df.index,
            sigma=df["laser_power_err"],
        )

        return (
            popt,
            df,
        )

    @staticmethod
    def linear_function(x, m, b):
        return x * m + b

    def laser_power(self, voltage: float) -> float:
        if -0.750 < voltage > 0.0:
            raise ValueError(f"Cannot predict laser power for {voltage} mV")
        popt, _ = self._fit_photo_voltage_to_discommoding_cooling_power
        print(f"{voltage * 1000} mV converts to {self.linear_function(voltage, *popt) * 1000} mW")
        return self.linear_function(voltage, *popt)

    @property
    def cooldown_data_indexed_by_time(self) -> [pd.DataFrame]:
        return [self.evaluate_cooldown(cooldown_df.data) for cooldown_df in self.cooldowns]

    def evaluate_cooldown(self, df: pd.DataFrame) -> pd.DataFrame:
        # recalculate the index to not have date times but seconds starting from 0
        try:
            df.set_index("_time", inplace=True)
        except KeyError as exc:
            print(f"{df}")
            raise KeyError(f"Key _time not found. Only have {df.columns}") from exc
        df["seconds"] = (pd.to_datetime(df.index.to_series()) - df.index[0]).dt.total_seconds()

        df.set_index("seconds", inplace=True)

        si = materials.silicon.Silicon()
        time_series = df.index.to_series()
        try:
            df["_value"]
        except KeyError:
            df["_value"] = df["Temperature_Testmass_(K)"]
        if self.temperatures_need_correction:
            df["_value"] = df["_value"].apply(corrector.testmass_sensor.correct_temperature)
        df["_value"] = df["_value"].rolling(60).mean()

        df["specific heat"] = si.specific_heat(df["_value"])
        df["temperature derivated"] = np.gradient(df["_value"], time_series)
        # df["temperature derivated"] = df["temperature derivated"].rolling(window=10).median()
        # calculate the heating power
        df["heating power (W)"] = (
            -TM.mass
            * (
                df["specific heat"] * df["temperature derivated"]
                + df["_value"]
                * df["temperature derivated"]
                * si.specific_heat_derivated(df["_value"])
            )
            .rolling(window=10)
            .mean()
        ) - 0.0018121

        return df[df.index != np.NaN]

    def plot_by_time(self):
        dfs = self.cooldown_data_indexed_by_time
        for df in dfs:
            df["temperature derivated"].plot()
        plt.savefig("figures/time/temp-derivate.pdf")
        plt.close()

        # df["heating power (W)"].plot(
        #     ylabel="Discommonding Cooling Power (W)"
        # ).figure.savefig("figures/time/discommonding-heat-power.pdf")
        # plt.close()
        # df["specific heat"].plot().figure.savefig("figures/time/specific-heat.pdf")
        # plt.close()
        #
        # df["_value"].plot().figure.savefig("figures/time/cooldown-temp-time.pdf")
        # plt.close()
