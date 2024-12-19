from typing import Union
import numpy as np
import pandas as pd
import scipy
import scipy.constants as const
from uncertainties import ufloat, umath

from . import calibration
import cooling.theory as gc
from . import influx, pressure
from .lakeshore import temp_sensor


def print_err(var: ufloat):
    print("########")
    for var, er in var.error_components().items():
        print(var.tag, er)
    print("########")


class ACMeasurement(influx.InfluxDataContainer):
    def __init__(
        self,
        start,
        stop,
        laser_calibration: Union[
            calibration.HighPressureCalibration,
            calibration.PowerMeterCalibration,
            calibration.MixedCalibration,
        ] = calibration.HighPressureCalibration,
        # heating_calibration: heating.Calibration = heating.PowerMeterCalibration,
        run_number: int = 0,
    ):
        super().__init__(start=start, stop=stop)
        self.run_number = run_number
        self.calibration = laser_calibration

        self.absorbed_laser_power = self.calibration.laser_power(
            voltage=ufloat(
                self.data["photo_voltage"].mean(),
                self.data["photo_voltage"].std(),
                tag="photodiode (statistic)",
            )
        )

        self.discommoding_cooling = (
            self.calibration.discommoding_cooling_power_by_delta_temperature(
                tm_temperature=self.data["temperatures"]["testmass"].mean(),
                frame_temperature=self.data["temperatures"]["holder"].mean(),
            )
        )

        self.gas_cooling = abs(self.absorbed_laser_power) - self.discommoding_cooling
        self.frame = gc.ThermalObject(
            temperature=ufloat(
                self.data["temperatures"]["holder"].mean(),
                self.data["temperatures"]["holder"].std(),
                tag="frame temp statistical",
            )
        )
        self.frame.temperature += ufloat(
            0,
            temp_sensor.systematical_error(self.data["temperatures"]["holder"].mean()),
            tag="frame temp systematical",
        )
        self.mirror = gc.TestMass(
            temperature=ufloat(
                self.data["temperatures"]["testmass"].mean(),
                self.data["temperatures"]["testmass"].std(),
                tag="tm temp statistical",
            ),
        )
        self.mirror.temperature -= ufloat(36e-3, 36e-3, tag="silicon temperature (systematic)")
        self.mirror.temperature += ufloat(
            0,
            temp_sensor.systematical_error(self.data["temperatures"]["testmass"].mean()),
            tag="tm temp systematical",
        )
        _, _, _, self.probability_pressure_stable, _ = scipy.stats.linregress(
            pd.to_datetime(self.data["pressure"].index).astype("int64") / 10**9,
            self.data["pressure"],
        )
        # print(f"{self.probability_pressure_stable=}")
        pressure_warm_mean = self.data["pressure"].mean()
        pressure_warm = ufloat(
            nominal_value=pressure_warm_mean,
            std_dev=self.data["pressure"].std(),
            tag="pressure (statistic)",
        )
        pressure_warm += ufloat(
            0,
            pressure_warm_mean
            * pressure.pfeiffer_relative_systematic_error(pressure=pressure_warm_mean),
            tag="pressure (systematic)",
        )

        self.gas_in = gc.Helium(
            pressure=0.5
            * pressure_warm
            * (self.frame.temperature / ufloat(294, 3, tag="lab temp")) ** 0.5,
            temperature=self.frame.temperature,
        )
        self.gas_out = gc.Helium(
            pressure=0.5
            * pressure_warm
            * (self.mirror.temperature / ufloat(294, 3, tag="lab temp")) ** 0.5,
            temperature=self.mirror.temperature,
        )

    def knudsen_number(self, accommodation_coefficient: float):
        """
        :return:
        """
        # lambda_in = self.gas_in.mean_free_path

        _sum_of_temps = self.frame.temperature + self.mirror.temperature
        product_of_temps = self.frame.temperature * self.mirror.temperature
        sqrt_of_temps = umath.sqrt(self.frame.temperature / self.mirror.temperature)

        v_in_in = v_out2_out2 = umath.sqrt(
            const.k * self.frame.temperature * (8 - np.pi) / self.gas_in.mass
        )
        v_out1_out1 = umath.sqrt(
            const.k * self.mirror.temperature * (8 - np.pi) / self.gas_out.mass
        )
        v_out1_out2 = v_out2_out1 = umath.sqrt(
            (const.k / self.gas_out.mass)
            * (4 * _sum_of_temps - np.pi * umath.sqrt(product_of_temps))
        )

        v_in_out1 = v_out1_in = umath.sqrt(
            (const.k / self.gas_out.mass)
            * (4 * _sum_of_temps + np.pi * umath.sqrt(product_of_temps))
        )
        v_in_out2 = v_out2_in = umath.sqrt(
            const.k * self.frame.temperature * (8 + np.pi) / self.gas_in.mass
        )

        v_in = v_out2 = umath.sqrt(
            9 * np.pi * const.k * self.gas_in.temperature / (8 * self.gas_in.mass)
        )
        v_out1 = umath.sqrt(
            9 * np.pi * const.k * self.gas_out.temperature / (8 * self.gas_out.mass)
        )

        lambda_in = (
            const.k
            * self.frame.temperature
            * v_in
            / (
                np.pi
                * self.gas_in.molecular_diameter**2
                * self.gas_in.pressure
                * (
                    v_in_in
                    + accommodation_coefficient * sqrt_of_temps * v_in_out1
                    + (1 - accommodation_coefficient) * v_in_out2
                )
            )
        )

        lambda_out_1 = (
            const.k
            * self.mirror.temperature
            * v_out1
            / (
                np.pi
                * self.gas_in.molecular_diameter**2
                * self.gas_in.pressure
                * (
                    accommodation_coefficient * sqrt_of_temps * v_out1_out1
                    + v_out1_in
                    + (1 - accommodation_coefficient) * v_out1_out2
                )
            )
        )
        lambda_out_2 = (
            const.k
            * self.frame.temperature
            * v_out2
            / (
                np.pi
                * self.gas_in.molecular_diameter**2
                * self.gas_in.pressure
                * (
                    accommodation_coefficient * sqrt_of_temps * v_out2_out1
                    + v_out2_in
                    + (1 - accommodation_coefficient) * v_out2_out2
                )
            )
        )
        a = (
            lambda_in
            + accommodation_coefficient * sqrt_of_temps * lambda_out_1
            + (1 - accommodation_coefficient) * lambda_out_2
        )
        b = 2 + accommodation_coefficient * (sqrt_of_temps - 1)

        return (a / b) / self.mirror.distance_to_frame

    def _gas_cooling_in_free_molecular_flow(self) -> float:
        return self.gas_cooling

    def _gas_cooling_in_transitional_flow(self) -> float:
        viscous_cooling = self.gas_in.viscous_flow_cooling(
            surface=self.mirror.surface_total,
            distance=self.mirror.distance_to_frame,
            temp_hot_surface=self.mirror.temperature,
            temp_cold_surface=self.frame.temperature,
        )
        fmf_cooling = self.gas_cooling * viscous_cooling / abs(viscous_cooling - self.gas_cooling)
        # print(f"Viscous flow cooling of {viscous_cooling * 1000} mW ")
        # print(f"Fmf cooling of {fmf_cooling * 1000} mW ")
        # print(f"Gas cooling of {self.gas_cooling * 1000} mW")
        return fmf_cooling

    @property
    def ac(self) -> ufloat:
        # print(f"Pressure in on TM: {self.gas_in.pressure}")

        gas_cooling_power = self._gas_cooling_in_free_molecular_flow()
        factor = ((const.pi * self.gas_in.mass * self.frame.temperature) / (8 * const.k)) ** 0.5

        ac = (
            factor
            * gas_cooling_power
            / (
                self.gas_in.pressure
                * self.mirror.surface_total
                * (self.mirror.temperature - self.frame.temperature)
            )
        )

        # print(f"{self.gas_in.pressure=}")
        # print(f"{self.mirror.surface_total=}")
        # print(f"{self.mirror.temperature=}")
        # print(f"{self.frame.temperature=}")
        # print(f"Kn for alpha {ac}={self.knudsen_number(accommodation_coefficient=ac)}")

        if 8 < self.knudsen_number(accommodation_coefficient=ac) and self.mirror.temperature > 8:
            # self.knudsen_number(accommodation_coefficient=ac)=}")
            return ac
        print(
            f"rejecting {
                self.mirror.temperature} K because of Kn ={
                self.knudsen_number(
                    accommodation_coefficient=ac)}"
        )
        return ufloat(0, 0)

    def _download_data(self):
        _pressures = (
            influx.get_data(
                field="Pressure_Gas_handling_(mbar)",
                start=self.start,
                stop=self.stop,
            ).set_index("_time")
        )["Pressure_Gas_handling_(mbar)"]
        _pressures = (_pressures * 100).rename("Pressure_Gas_handling_(Pa)")

        return {
            "temperatures": {
                "testmass": (
                    influx.get_data(
                        field="Temperature_Testmass_(K)",
                        start=self.start,
                        stop=self.stop,
                    ).set_index("_time")
                )["Temperature_Testmass_(K)"],
                "holder": (
                    influx.get_data(
                        field="Temperature_Holver_(K)", start=self.start, stop=self.stop
                    ).set_index("_time")
                )["Temperature_Holver_(K)"],
                "steel": (
                    influx.get_data(
                        field="Temperature_Steel_(K)", start=self.start, stop=self.stop
                    ).set_index("_time")
                )["Temperature_Steel_(K)"],
            },
            "pressure": (
                (
                    influx.get_data(
                        field="Pressure_Gas_handling_(mbar)",
                        start=self.start,
                        stop=self.stop,
                    ).set_index("_time")
                )["Pressure_Gas_handling_(mbar)"]
                * 100
            ).rename("Pressure_Gas_handling_(Pa)"),
            "photo_voltage": (
                influx.get_data(
                    field="Voltage_Photodiode_(V)", start=self.start, stop=self.stop
                ).set_index("_time")
            )["Voltage_Photodiode_(V)"],
        }
