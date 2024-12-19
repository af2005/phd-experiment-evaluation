"""
The lakeshore controller unit translates the sensor resistances only approximately to temperatures,
although there is a better calibration known.
This corrector reverses the controller's calibration and applies the more sophisticated one.
"""

import datetime
from functools import cached_property
from typing import Iterable, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import PositiveFloat
from scipy import interpolate
from .. import influx as influx


class Cernox1070:
    def __init__(
        self,
        serial_number: str,
        zl,
        zu,
        resistence_thresholds,
        chebychev_coefficients,
        controller_calibration,
    ):
        self.serial_number = serial_number
        self.chebychev_coefficients = chebychev_coefficients
        self.zl = zl
        self.zu = zu
        self.thresholds = np.log10(resistence_thresholds)
        self._controller_calibration = self._interpolate_controller_calibration(
            controller_calibration
        )

    @staticmethod
    def _excitation_current(resistance: PositiveFloat):
        if resistance < 10:
            return 1e-3
        if resistance < 30:
            return 300e-6
        if resistance < 100:
            return 100e-6
        if resistance < 300:
            return 30e-6
        if resistance < 1e3:
            return 10e-6
        if resistance < 3e3:
            return 3e-6
        if resistance < 10e3:
            return 1e-6
        if resistance < 30e3:
            return 300e-9
        if resistance < 100e3:
            return 100e-9

    def excitation_current(self, resistance: Union[PositiveFloat, Iterable[PositiveFloat]]):
        try:
            return self._excitation_current(resistance)
        except ValueError:
            return [self._excitation_current(r) for r in resistance]

    def self_heating_power(self, controller_temperature):
        # oldTODO this should use controller_temperature and not correct_temperature
        resistance = 10 ** self.controller_temp_to_logresistence(controller_temperature)
        current = self.excitation_current(resistance)
        return resistance * current * current

    @staticmethod
    def _interpolate_controller_calibration(data: pd.DataFrame):
        return interpolate.interp1d(data["Temperature"], data["Units"])

    def temperature(self, log_resistence: Union[float, np.array]) -> Union[float, np.array]:
        if isinstance(log_resistence, float):
            log_resistence = [log_resistence]

        output = []
        for res in log_resistence:
            _mask = sum((threshold > res for threshold in self.thresholds)) - 1

            k = ((res - self.zl[_mask]) - (self.zu[_mask] - res)) / (
                self.zu[_mask] - self.zl[_mask]
            )
            output.append(
                sum(
                    (
                        c * np.cos(i * np.arccos(k))
                        for i, c in enumerate(self.chebychev_coefficients[_mask])
                    )
                )
            )
        if isinstance(log_resistence, float):
            return output[0]
        return np.array(output)

    def controller_temp_to_logresistence(self, temperature: Union[float, Iterable[float]]):
        return self._controller_calibration(temperature)

    def correct_temperature(self, controller_temperature):
        return self.temperature(self.controller_temp_to_logresistence(controller_temperature))


try:
    tm_data = pd.read_csv("experiment/lakeshore/X166120.csv", index_col="No.")
    steel_data = pd.read_csv(
        "experiment/lakeshore/X166521.csv",
        header=6,
        sep="\s+",
        index_col="No.",
    )
    holder_data = (
        pd.read_csv("experiment/lakeshore/X166410.csv", header=6, index_col="No.", sep="\s+"),
    )
except FileNotFoundError:
    tm_data = pd.read_csv("X166120.csv", index_col="No.")
    steel_data = pd.read_csv(
        "X166521.csv",
        header=6,
        sep="\s+",
        index_col="No.",
    )
    holder_data = pd.read_csv("X166410.csv", header=6, index_col="No.", sep="\s+")

testmass_sensor = Cernox1070(
    serial_number="X166120",
    zl=[2.9865155348, 2.29616496112, 1.86735704361],
    zu=[4.0423673708, 3.10026807488, 2.44001638189],
    resistence_thresholds=[9193, 1097, 236.1, 74.83],
    chebychev_coefficients=[
        [
            11.954033,
            -11.228549,
            3.544612,
            -0.776604,
            0.097848,
            0.001971,
            -0.002321,
            -0.000558,
        ],
        [
            64.399486,
            -52.968968,
            11.083659,
            -1.472137,
            0.128607,
            -0.001411,
            0.002282,
            -0.001944,
        ],
        [
            194.127193,
            -114.959911,
            18.021446,
            -2.455035,
            0.403968,
            -0.065414,
            0.010503,
            -0.002319,
        ],
    ],
    controller_calibration=tm_data,
)

steel_sensor = Cernox1070(
    serial_number="X166521",
    zl=[3.12381698012, 2.27918611887, 1.86307587156],
    zu=[4.35661543513, 3.23373459733, 2.4046626619],
    resistence_thresholds=[1.849e4, 1505, 217.8, 73.87],
    chebychev_coefficients=[
        [
            12.826711,
            -12.608974,
            4.253665,
            -1.014494,
            0.147290,
            -0.002116,
            -0.002821,
            -0.000771,
        ],
        [
            84.655877,
            -75.328938,
            17.227128,
            -2.452741,
            0.232452,
            -0.026534,
            0.013013,
            -0.004004,
        ],
        [261.092935, -139.567133, 20.966304, -2.968714, 0.466202, -0.075630, 0.010514],
    ],
    controller_calibration=steel_data,
)

holder_sensor = Cernox1070(
    serial_number="X166410",
    zl=[3.1419207664, 2.42140530881, 1.96750155815],
    zu=[4.28239777418, 3.25978161477, 2.57170119101],
    resistence_thresholds=[1.572e4, 1573, 317.6, 94.31],
    chebychev_coefficients=[
        [
            11.818687,
            -11.094717,
            3.592858,
            -0.830155,
            0.118092,
            -0.001314,
            -0.003263,
            -0.000262,
        ],
        [
            64.293687,
            -52.945875,
            11.108112,
            -1.445294,
            0.119917,
            -0.000471,
            0.001745,
            -0.001392,
        ],
        [194.456927, -115.189789, 17.609611, -2.264124, 0.359759, -0.053507, 0.005423],
    ],
    controller_calibration=pd.read_csv(
        "experiment/lakeshore/X166410.csv",
        header=6,
        sep="\s+",
        index_col="No.",
    ),
)


class SecondaryCorrector:
    """
    This class corrects the tm temp sensor to the correct value, based on our final sensor comparison
    """

    @cached_property
    def _fit_cooling_phase(self):
        start = "2023-06-22T08:00:00Z"
        stop = "2023-06-24T10:30:00Z"
        return self._fit_secondary_tm_correction(start=start, stop=stop)

    @cached_property
    def _fit_heating_phase(self):
        start = "2023-06-27T11:50:00Z"
        stop = "2023-06-27T13:30:00Z"
        return self._fit_secondary_tm_correction(start=start, stop=stop)

    def _fit_secondary_tm_correction(self, start, stop):
        temperature_tm = influx.InfluxDataContainer(
            start=start,
            stop=stop,
            field="Temperature_Testmass_(K)",
            second_correction=False,
        ).data
        temperature_holver = influx.InfluxDataContainer(
            start=start,
            stop=stop,
            field="Temperature_Holver_(K)",
            second_correction=False,
        ).data
        temperature_steel = influx.InfluxDataContainer(
            start=start,
            stop=stop,
            field="Temperature_Steel_(K)",
            second_correction=False,
        ).data
        times_holver = [
            datetime.datetime.timestamp(time) * 1000 for time in temperature_holver["_time"]
        ]
        times_steel = [
            datetime.datetime.timestamp(time) * 1000 for time in temperature_steel["_time"]
        ]
        times_tm = [datetime.datetime.timestamp(time) * 1000 for time in temperature_tm["_time"]]

        temperature_holver_interpolated = interpolate.interp1d(
            times_holver,
            temperature_holver["Temperature_Holver_(K)"],
            fill_value="extrapolate",
        )
        temperature_steel_interpolated = interpolate.interp1d(
            times_steel,
            temperature_steel["Temperature_Steel_(K)"],
            fill_value="extrapolate",
        )

        correct_tm_temp_data = [
            (temperature_holver_interpolated(time) + temperature_steel_interpolated(time)) / 2
            for time in times_tm
        ]
        return interpolate.interp1d(
            temperature_tm["Temperature_Testmass_(K)"], correct_tm_temp_data
        )

    def correct_v2_temperature(self, incorrect_temperature, phase="manual"):
        # return
        if phase == "manual":
            return incorrect_temperature - 33e-3
        if phase == "cooling":
            return self._fit_cooling_phase(incorrect_temperature)
        if phase == "heating":
            return self._fit_heating_phase(incorrect_temperature)


def plot_test_mass_sensor_characteristics():
    fig1, axes = plt.subplots()
    fig1.set_figheight(2.3)
    fig1.set_figwidth(3.5)

    print(testmass_sensor.self_heating_power(controller_temperature=6.861258))

    temps = np.linspace(6, 50, num=1000)
    resistances = 10 ** testmass_sensor.controller_temp_to_logresistence(temps)

    axes.plot(temps, resistances / 1000, label="Resistance (kΩ)")
    axes.plot(
        temps,
        [current * 1e6 for current in testmass_sensor.excitation_current(resistances)],
        label="Current (uA)",
    )
    axes.plot(
        temps,
        [
            voltage * 1e3
            for voltage in resistances * testmass_sensor.excitation_current(resistances)
        ],
        label="Voltage (mA)",
    )
    axes.plot(
        temps,
        [
            power * 1e9
            for power in resistances
            * testmass_sensor.excitation_current(resistances)
            * testmass_sensor.excitation_current(resistances)
        ],
        label="Self heating (nW)",
    )
    axes.set_yscale("log")
    axes.text(x=30, y=1, s="Resistance (kΩ)", c="C0")
    axes.text(x=30, y=12, s="Current (μA)", c="C1")
    axes.text(x=30, y=3.5, s="Voltage (mA)", c="C2")
    axes.text(x=30, y=32, s="Self heating (nW)", c="C3")
    axes.set_xlabel("Temperature (K)")

    fig1.savefig(
        "figures/tm_sensor_heating.pgf",
        backend="pgf",
    )
    plt.close()


if __name__ == "__main__":
    print(10 ** testmass_sensor.controller_temp_to_logresistence(4))
    # c = SecondaryCorrector()
