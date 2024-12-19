import requests

API_TOKEN = "hidden"


class Connector:
    def __init__(self):
        pass

    @staticmethod
    def _get_table(table_number: int):
        return requests.get(
            f"https://database1.lasnq.physnet.uni-hamburg.de/api/database/rows/table/{table_number}/?"
            f"user_field_names=true&size=200",
            headers={"Authorization": f"Token {API_TOKEN}"},
        ).json()["results"]

    def cooldowns(self):
        return self._get_table(table_number=349)

    def low_pressure_tm_temperature_correction(self):
        return self._get_table(table_number=696)

    def discommoding_cooling_data(self):
        return (
            self._get_table(table_number=344)
            + self._get_table(table_number=688)
            + self._get_table(687)
        )

    def base_heating_H0(self):
        return self._get_table(table_number=693)

    def base_heating_L0(self):
        raise NotImplementedError
        # return self._get_table(table_number=693)

    def base_heating_F0(self):
        return self._get_table(table_number=694)

    def pd_calibration_at_high_pressure(self):
        return self._get_table(table_number=692)

    def accommodations(self):
        return self._get_table(table_number=348)
