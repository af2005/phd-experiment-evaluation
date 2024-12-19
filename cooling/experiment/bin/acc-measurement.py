import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.odr import ODR, Model, RealData
from uncertainties import ufloat

from cooling.experiment.evaluation import accommodation, baserow, calibration, fitting
from cooling.theory.objects import TestMass
import helpers


def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):
    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0, 1, nc))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    cols = np.zeros((nc * nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = mpl.colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv, nsc).reshape(nsc, 3)
        arhsv[:, 1] = np.linspace(chsv[1], 0.25, nsc)
        arhsv[:, 2] = np.linspace(chsv[2], 1, nsc)
        rgb = mpl.colors.hsv_to_rgb(arhsv)
        cols[i * nsc : (i + 1) * nsc, :] = rgb

    return cmap


c1 = categorical_cmap(4, 3, cmap="tab10")


# mpl.rcParams["axes.prop_cycle"] = more_colors_cycle


def acc_from_baserow():
    connector = baserow.Connector()
    accs = connector.accommodations()
    return [(a["Start"], a["Stop"], a["Run #"]) for a in accs if a["Active"]]


def acc_measurements():
    laser_calibration = calibration.HighPressureCalibration()
    return [
        accommodation.ACMeasurement(
            start=t[0], stop=t[1], run_number=t[2], laser_calibration=laser_calibration
        )
        for t in acc_from_baserow()
    ]


def tex_round(number, digits=3):
    number, power = "{:e}".format(number).split("e")
    number = round(float(number), digits)
    power = int(power)
    print(number, power)
    if power == 0:
        return f"{number}$&$"
    # number = "{:.3e}".format(number).replace("e", "\cdot 10^{")
    return f"{number}\,\cdot\,10^" + "{" + str(power) + "}"


if __name__ == "__main__":
    # heating_calibration_v2 = experiment.heating.LowPressureCalibration(
    #     start="2023-01-11T09:00:00Z",
    #     stop="2023-01-11T17:00:00Z",
    #     field="Temperature_Testmass_(K)",
    #     measurement="live",
    # )
    helpers.figures.initialize()

    print(TestMass(temperature=10).diameter)
    print(TestMass(temperature=10).thickness)

    measurements = acc_measurements()

    acs = []
    acs_error = []
    temps = []
    temps_error = []
    colors = []
    annotations = []
    alphas = []
    measurements = sorted(
        measurements, key=lambda d: d.data["temperatures"]["testmass"].mean()
    )
    all_uncertainty_components = pd.DataFrame()
    laser_power_fluctations = []

    pd_sample_data = []
    for m in measurements:
        if (
            m.probability_pressure_stable > 1e-30
            and m.ac.nominal_value > 0
            and 11 < m.data["temperatures"]["testmass"].mean()
        ):
            print("")
            print("=======================")
            print(f"Measurement {m.start[8:10]}.{m.start[5:7]} {m.start[11:13]}h")
            print("")
            print(f'TM temp {m.data["temperatures"]["testmass"].mean()}')
            # print(f'Pressure warm {m.data["pressure"].mean()}')
            # print(f"Pressure cold {2 * m.gas_in.pressure}")
            # print(f"Pressure stability {m.probability_pressure_stable}")
            print(f"Heating power {m.absorbed_laser_power=}")
            laser_power_fluctations.append(
                m.absorbed_laser_power.std_dev / m.absorbed_laser_power.nominal_value
            )
            print(f"Discommoding cooling power {m.discommoding_cooling=}")

            acs.append(m.ac.nominal_value)
            acs_error.append(m.ac.std_dev)
            annotations.append(m.start)

            temps.append(m.data["temperatures"]["testmass"].mean())
            temps_error.append(m.data["temperatures"]["testmass"].std())
            knudsen_number = m.knudsen_number(accommodation_coefficient=m.ac)
            # if knudsen_number < 7:
            #     raise ArithmeticError("We should not plot values with kn < 8")
            # if 7 < knudsen_number < 8:
            #     colors.append("C3")
            # if 8 < knudsen_number < 9:
            #     colors.append("C2")
            # elif 9<= knudsen_number < 10:
            #     colors.append("C1")
            # elif 10 <= knudsen_number:
            #     colors.append("C0")
            colors.append(f"C{int(m.run_number) - 1}")
            alphas.append(max(1, m.probability_pressure_stable))
            print(
                f"ac={m.ac} @ "
                f'{ufloat(m.data["temperatures"]["testmass"].mean(), m.data["temperatures"]["testmass"].std())} K'
                f" with a gas cooling of {(m.gas_cooling * 1000)} mW"
            )

            print(m.data["photo_voltage"].std())
            print(f"{knudsen_number=}")
            if 18 < m.data["temperatures"]["testmass"].mean() < 20:
                first_date = m.data["photo_voltage"][0]
                pd_sample_data.append(m.data["photo_voltage"])

            print("-----------Uncertainties--------------")
            uncertainty_components = pd.DataFrame(
                m.ac.error_components().items(),
                columns=["error var", "value"],
            )
            uncertainty_components["variances"] = uncertainty_components["value"] ** 2
            uncertainty_components["uncertainty source"] = uncertainty_components[
                "error var"
            ].apply(lambda x: x.tag)
            uncertainty_components = uncertainty_components.set_index(
                "uncertainty source"
            )
            uncertainty_components = uncertainty_components["variances"]
            uncertainty_components = uncertainty_components.groupby(level=0).sum()
            for variance in [
                "tm temp statistical",
                "tm temp systematical",
                "mirror diameter",
                "mirror thickness",
                "lab temp",
                "pd calibration intercept",
                "pd calibration slope",
                "frame temp statistical",
                "frame temp systematical",
            ]:
                try:
                    uncertainty_components["other"] += uncertainty_components[variance]
                except KeyError:
                    uncertainty_components["other"] = uncertainty_components[variance]
                uncertainty_components.drop(variance, inplace=True)

            print(uncertainty_components)

            all_uncertainty_components[m.mirror.temperature.n] = (
                uncertainty_components.T
            )

            if 19.63 < m.data["temperatures"]["testmass"].mean() < 19.64:
                pd.concat([all_uncertainty_components, uncertainty_components], axis=0)
                print(m.ac.error_components())
                for var, error in m.ac.error_components().items():
                    uncertainty_components[var] = error
                    print("{}, {}".format(var.tag, np.sqrt(error)))
            # print("______________________________________")
        else:
            pass
            # print(
            #     f"Rejecting {m.start[8:10]}.{m.start[5:7]} {m.start[11:13]}h. "
            #     f"Pressure stab: {m.probability_pressure_stable}. "
            #     f"Knudsen {m.knudsen_number(accommodation_coefficient=0.3)}"
            #     f""
            # )

    # -----------------------------------
    # uncertainty plot
    # -----------------------------------
    uncertainty_plot = all_uncertainty_components.T.plot.bar(
        logy=False,
        stacked=True,
        color=[
            "#1984c5",
            "#a7d5ed",
            "#e0ae9d",
            "#de6e56",
            "#c22f1f",
            "#e2e2e2",
        ],  # color=more_colors_cycle
    )
    uncertainty_plot.set_xticklabels(
        [round(float(x.get_text()), 1) for x in uncertainty_plot.get_xticklabels()],
        fontsize=8,
    )
    uncertainty_plot.legend().set_title("")
    xlabel = uncertainty_plot.set_xlabel("Silicon temperature $T$ (K)")
    ylabel = uncertainty_plot.set_ylabel("Variance of $\\alpha$")
    fig = uncertainty_plot.get_figure()
    fig.set_figheight(helpers.figures.set_size()[1])
    fig.set_figwidth(helpers.figures.set_size()[0])

    plt.gca().ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # formatter = mticker.ScalarFormatter(useMathText=True, )
    # plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().yaxis.get_major_formatter().set_scientific(True)

    fig.savefig(
        "figures/uncertainties.pdf",
        backend="pgf",
        # bbox_inches="tight",
    )
    fig.savefig(
        "figures/uncertainties.pgf",
        backend="pgf",
        # bbox_inches="tight",
    )

    # --------------------------------------
    # Accommodation plot
    # --------------------------------------
    fig1, ax = plt.subplots(figsize=helpers.figures.set_size())
    ax.errorbar(x=[], y=[], yerr=[], color="black", label="Measured data", fmt=".")

    for i in range(len(temps)):
        if not colors[i]:
            colors[i] = 2

        error_linewidth = 0.5 if temps[i] < 16 else 1.5
        size = "," if temps[i] < 16 else "."
        alpha = 0.4 if temps[i] < 10 else alphas[i]
        ax.errorbar(
            x=temps[i],
            y=acs[i],
            xerr=temps_error[i],
            yerr=acs_error[i],
            marker=".",
            # elinewidth = error_linewidth,
            color=colors[i],
            alpha=alpha,
        )
    for count, annotation in enumerate(annotations):
        # ax.text(
        #     temps[count] * 1.003,
        #     acs[count] * 1.0004,
        #     annotation[8:10] + "." + annotation[5:7] + " " + annotation[11:13] + "h ",
        #     rotation=75,
        #     size=4,
        # )
        pass

    # ---------------------------------------
    # Acc fit
    # ---------------------------------------

    model = Model(fitting.sigmuid)
    real_data = RealData(x=temps, sx=temps_error, y=acs, sy=acs_error)
    odr = ODR(
        real_data,
        model,
        # beta0=[4.215, -8.505e-1, 8.27e-2, -3.964e-3, 9.291e-5, -8.531e-7],
        beta0=[0.3, -1, 0, 0.7],
    )
    fit = odr.run()

    with open("acc-fit.tex", "w+") as texfile:
        # Writing data to a file
        #                   &       \\
        texfile.writelines(f"$a={tex_round(fit.beta[0], 2)},")
        texfile.writelines(f"b={tex_round(fit.beta[1], 2)}" + "\\,\\mathrm{K}^{-1},")
        texfile.writelines(f"c={tex_round(fit.beta[2], 2)}" + "\\,\\mathrm{K}$\\@.")
        # texfile.writelines(
        #     f"d&${tex_round(fit.beta[3], 2)}" + "\\mathrm{K}^{-3}$ \\\\ "
        # )
        # texfile.writelines(
        #     f"e&${tex_round(fit.beta[4], 2)}" + "\\mathrm{K}^{-4}$ \\\\ "
        # )
        # texfile.writelines(f"f&${tex_round(fit.beta[5], 2)}" + "\\mathrm{K}^{-5}$")

    spline, fp, ier, msg = interpolate.splrep(
        x=temps, y=acs, w=[1 / err for err in acs_error], k=5, full_output=True
    )
    spline_x = np.linspace(10, 32)
    # print(f"{spline=}")
    # print(f"{fp=}")
    # print(f"{ier=}")
    # print(f"{msg=}")
    # plt.plot(spline_x, interpolate.splev(spline_x, spline))

    plt.plot(
        spline_x,
        model.fcn(fit.beta, spline_x),
        color="C1",
        label="Fit",
    )
    # plt.fill_between(spline_x, polynom_fifth_grade(spline_x, *(curve_fit+perr)),
    # polynom_fifth_grade(spline_x, *curve_fit))

    xlabel = plt.xlabel("Silicon temperature $T$ (K)")
    ylabel = plt.ylabel("Accommodation coefficient of helium $\\alpha$")
    plt.legend()
    plt.minorticks_on()
    plt.xlim(11, 31)
    plt.xticks([12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
    plt.grid(which="major", axis="x")
    plt.grid(which="major", axis="y")
    # plt.grid()

    # knudsen_above_10 = mpatches.Patch(color='C0', label='Kn > 10')
    # knudsen_9_10 = mpatches.Patch(color='C1', label='10 > Kn > 9')
    # knudsen_8_9 = mpatches.Patch(color='C2', label='9 > Kn > 8')
    # knudsen_7_8 = mpatches.Patch(color='C3', label='8 > Kn > 7')
    # plt.legend(handles=[knudsen_above_10, knudsen_9_10, knudsen_8_9, knudsen_7_8])
    plt.savefig("figures/ac.pdf")
    plt.savefig(
        "figures/ac.pgf",
        # bbox_extra_artists=[xlabel, ylabel],
        # bbox_inches="tight",
    )
    # import tikzplotlib
    #
    # tikzplotlib.save("figures/tikz/ac.tex")
