import math
import random

import numpy as np
import scipy.constants as const
import scipy.special
import scipy.stats


def biased_coin_toss(probability_for_true):
    return random.random() < probability_for_true


class KnudsenGen2D(scipy.stats.rv_continuous):
    def __init__(self, pdf=None):
        super().__init__()
        self.custom_pdf = pdf

    def _pdf(self, x, *args):
        if x < -math.pi / 2:
            return 0.0
        if x > math.pi / 2:
            return 0.0
        return 0.5 * np.cos(x)

    def _cdf(self, x, *args):
        if x < -math.pi / 2:
            return 0.0
        if x > math.pi / 2:
            return 1.0
        return 0.5 * (np.sin(x))

    def _ppf(self, q, *args):
        return 2 * np.arcsin(q)


class KnudsenGen3D(scipy.stats.rv_continuous):
    def __init__(self, pdf=None):
        super().__init__()
        self.custom_pdf = pdf

    def _pdf(self, x, *args):
        if x < -math.pi / 2:
            return 0.0
        if x > math.pi / 2:
            return 0.0
        return np.cos(x) / math.pi

    def _cdf(self, x, *args):
        if x < -math.pi / 2:
            return 0.0
        if x > math.pi / 2:
            return 1.0
        return (np.sin(x)) / math.pi

    def _ppf(self, q, *args):
        return math.pi * np.arcsin(q)


class Maxwell3D(scipy.stats.rv_continuous):
    def __init__(self, pdf=None, characteristic_thermal_velocity=1):
        super().__init__()
        self.custom_pdf = pdf
        self.characteristic_thermal_velocity = characteristic_thermal_velocity
        self.b = 2 * self.characteristic_thermal_velocity**2  # 2v^2

    def _pdf(self, x, *args):
        if x < 0:
            return 0.0
        return x**3 / (2 * self.characteristic_thermal_velocity**4) * np.exp(-(x**2) / self.b)

    def _cdf(self, x, *args):
        if x < 0:
            return 0.0
        return 1 + np.exp(-(x**2) / self.b) * (-(x**2) / self.b - 1)

    def _ppf(self, x, *args):
        w = scipy.special.lambertw(((x - 1) / const.e), k=-1)
        return np.real(np.sqrt(-self.b * (w + 1)))


class DiffuseEmittedParticleVelocity(scipy.stats.rv_continuous):
    def __init__(self, mass, temperature, pdf=None):
        super().__init__()
        self.custom_pdf = pdf
        self.mass = mass
        self.temperature = temperature

    def _pdf(self, x, *args):
        kbT = const.k * self.temperature
        pre_factor = np.sqrt(2 / np.pi * ((self.mass / kbT) ** 3))
        return pre_factor * x * x * np.exp(-self.mass * x * x / (2 * kbT))
