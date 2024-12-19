import pickle

import influxdb_client as influx
import pandas as pd
from influxdb_client import Dialect

from .lakeshore.corrector import (
    SecondaryCorrector,
    holder_sensor,
    steel_sensor,
    testmass_sensor,
)

TOKEN = "hidden"
ORG = "LasNQ"
BUCKET = "Helium Accommodation"

CLIENT = influx.InfluxDBClient(
    url="https://data.lasnq.physnet.uni-hamburg.de", token=TOKEN, org=ORG
)

api = CLIENT.query_api()
SECONDARY_CORRECTOR = SecondaryCorrector()


def get_test_data():
    tables = api.query_csv(
        'from(bucket:"Helium Accommodation") |> range(start: -1m)',
        dialect=Dialect(
            header=False,
            delimiter=",",
            comment_prefix="#",
            annotations=[],
            date_time_format="RFC3339",
        ),
    )
    return tables


def get_date(query):
    return api.query_data_frame(query)


def get_data(field, start, stop, measurement="live", second_correction=True) -> pd.DataFrame:
    print(f"Downloading {field} from influx {start=} and {stop=}")
    q = (
        f'from(bucket:"Helium Accommodation") '
        f"|> range(start: {start}, stop: {stop}) "
        f'|> filter(fn: (r) => r["_measurement"] == "{measurement}")'
        "|> filter("
        "fn: (r) => "
        f'r["_field"] == "{field}"'
        ")"
        f'|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'
    )
    df = api.query_data_frame(q)
    if stop < "2023-03-28T11:47:00Z":
        # temp values before that were with the lakeshore controller calibration, which is inaccurate
        if field == "Temperature_Testmass_(K)":
            df["Temperature_Testmass_(K)"] = testmass_sensor.correct_temperature(
                controller_temperature=df["Temperature_Testmass_(K)"]
            )

        if field == "Temperature_Holver_(K)":
            df["Temperature_Holver_(K)"] = holder_sensor.correct_temperature(
                controller_temperature=df["Temperature_Holver_(K)"]
            )
        if field == "Temperature_Steel_(K)":
            df["Temperature_Steel_(K)"] = steel_sensor.correct_temperature(
                controller_temperature=df["Temperature_Steel_(K)"]
            )

    # Currently I do think that we have a base heating and not a wrong correction
    # if field == "Temperature_Testmass_(K)" and second_correction:
    #     df["Temperature_Testmass_(K)"] = SECONDARY_CORRECTOR.correct_temperature(
    #         df["Temperature_Testmass_(K)"]
    #     )

    return df


def get_aggregated_data(
    field,
    start,
    stop,
    aggregation_window,
    measurement="live",
):
    print("Downloading aggregated data from influx")
    try:
        q = (
            f'from(bucket:"Helium Accommodation") '
            f"|> range(start: {start}, stop: {stop}) "
            f'|> filter(fn: (r) => r["_measurement"] == "{measurement}")'
            f'|> filter(fn: (r) => r["_field"] == "{field}")'
            f"|> aggregateWindow(every: {aggregation_window}, fn: mean, createEmpty: false)"
            f'|> yield(name: "mean")'
            f'|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'
        )
        return api.query_data_frame(q)[0]
    except KeyError:
        raise ValueError(f"No data returned for {start=} and {stop=} ")


class InfluxDataContainer:
    def __init__(
        self,
        start=None,
        stop=None,
        field="Temperature_Testmass_(K)",
        measurement="live",
        second_correction=True,
    ):
        self.start = start
        self.stop = stop
        self.field = field
        self.measurement = measurement
        self._data = None
        self.second_correction = second_correction

    @property
    def filename(self):
        return (
            (str(self.start) + str(self.stop) + str(self.field)).replace("-", "").replace(":", "")
        )

    def _save_data_to_file(self):
        pickle.dump(self._data, open(f"pickle/{self.filename}", "wb+"))

    def _read_data_from_file(self) -> pd.DataFrame:
        return pickle.load(open(f"pickle/{self.filename}", "rb"))

    @property
    def data(self) -> pd.DataFrame:
        if self._data is not None:
            return self._data
        try:
            self._data = self._read_data_from_file()
        except FileNotFoundError:
            print("file not found")
            self._data = self._download_data()
            self._save_data_to_file()
        return self._data

    def _download_data(self) -> pd.DataFrame:
        data = get_data(
            field=self.field,
            start=self.start,
            stop=self.stop,
            measurement=self.measurement,
            second_correction=self.second_correction,
        )
        return data
