from math import log10 as lg


class Polyamide:
    @staticmethod
    def thermal_conductivity(temperature):
        """
        From https://trc.nist.gov/cryogenics/materials/Polyamide%20(Nylon)/PolyamideNylon_rev.htm
        :param temperature:
        :return:
        """
        a = -2.6135
        b = 2.3239
        c = -4.7586
        d = 7.1602
        e = -4.9155
        f = 1.6324
        g = -0.2507
        h = 0.0131
        logT = lg(temperature)
        log_k = (
            a
            + b * logT
            + c * logT**2
            + d * logT**3
            + e * logT**4
            + f * logT**5
            + g * logT**6
            + h * logT**7
        )
        return 10**log_k


if __name__ == "__main__":
    p = Polyamide()
    print(p.thermal_conductivity(7.4))
