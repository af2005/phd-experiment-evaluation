import pandas as pd

from cooling.experiment.evaluation import influx

COOLDOWNS = [
    # ("2022-10-12T11:40:00Z", "2022-10-14T23:19:23Z", "T-testmass", "second run"),
    (
        "2023-01-11T11:30:00Z",
        # "2023-01-12T07:22:00Z",
        "2023-01-13T08:00:00Z",
        "Temperature_Testmass_(K)",
        "live",
        "jan 12, vacuum, pumping on",
        0,
    ),
    (
        "2023-02-06T11:00:00Z",
        "2023-02-08T09:02:00Z",
        "Temperature_Testmass_(K)",
        "live",
        "feb 06, vacuum, no pumping",
        -1.65 * 60,
    ),
    (
        "2023-02-10T13:18:00Z",
        "2023-02-11T12:35:00Z",
        "Temperature_Testmass_(K)",
        "live",
        "feb 10, vacuum, no pumping",
        28.2 * 60,
    ),
    (
        "2023-02-08T18:05:00Z",
        "2023-02-09T14:25:00Z",
        "Temperature_Testmass_(K)",
        "live",
        "feb 08, vacuum, no pumping",
        34.25 * 60,
    ),
    (
        "2023-02-01T17:30:00Z",
        "2023-02-02T09:50:00Z",
        "Temperature_Testmass_(K)",
        "live",
        "feb 01, p = 1.2e-4 mbar, no pumping",
        40 * 60,
    ),
    (
        "2023-03-16T18:00:00Z",
        "2023-03-17T19:30:00Z",
        "Temperature_Testmass_(K)",
        "live",
        "mar 16, Holder 8K, pumping on",
        26 * 60,
    ),
]


def shift_fit(x, a):
    return x + a


def plot_cooldown(dataframe: pd.DataFrame, label: str = "", shift=0):
    dataframe.index = dataframe.index + shift
    dataframe.index = dataframe.index / 60 / 60
    dataframe.index.name = "time (hours)"
    try:
        ax = dataframe["_value"].plot(ylabel="test mass temperature (K)", label=label)
    except KeyError:
        print(dataframe)
    minimum = dataframe["_value"].max()
    return minimum, ax.get_figure()


if __name__ == "__main__":
    minimums = []
    for cooldown in COOLDOWNS:
        data = influx.get_aggregated_data(
            field=cooldown[2],
            start=cooldown[0],
            stop=cooldown[1],
            measurement=cooldown[3],
            aggregation_window="30s",
        )
        minimum, fig = plot_cooldown(data, shift=cooldown[5], label=cooldown[4])

    fig.legend()
    fig.savefig("figures/cooldown.pdf")
