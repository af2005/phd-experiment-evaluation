import logging

import matplotlib.pyplot as plt
import numpy as np

logging.getLogger("matplotlib.font_manager").disabled = True


def initialize():
    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        "color",
        [
            "#111111",
            "#656fc5",
            "#e5515d",
            "#72D88C",
            "#e69f00",
            "#48A8C1",
            "#908d9d",
            "#53653c",
            "#ab6b49",
            "#a947c3",
        ],
        # ["#444444", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
    )
    import matplotlib.font_manager

    # matplotlib.font_manager.findSystemFonts(fontpaths=["/Users/afranke/Library/Fonts"], fontext='otf')
    # For pdf export
    font_path = (
        # "/Users/afranke/repos/gitlab.alexfranke.de/thesis/dthesis/fonts/source/sans"
        "/Users/afranke/repos/gitlab.alexfranke.de/thesis/doctoral-thesis/fonts/source/sans"
        "/SourceSans3-Regular.otf"
    )
    matplotlib.font_manager.fontManager.addfont(font_path)
    prop = matplotlib.font_manager.FontProperties(fname=font_path)

    matplotlib.use("pgf")
    plt.rcParams["pgf.texsystem"] = "xelatex"
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams["figure.constrained_layout.w_pad"] = 0.01
    plt.rcParams["figure.constrained_layout.h_pad"] = 0.04

    pgf_with_custom_preamble = {
        "font.family": "sans-serif",  # use serif/main font for text elements
        "font.size": 11,
        "font.sans-serif": prop.get_name(),
        "text.usetex": True,  # use inline math for ticks
        "pgf.rcfonts": True,  # don't setup fonts from rc parameters
        "pgf.preamble": r"\usepackage{unicode-math}"  # unicode math setup
        r"\renewcommand*\familydefault{\sfdefault}"
        r"\renewcommand*\mathdefault{\mathsf}",
    }
    matplotlib.rcParams.update(pgf_with_custom_preamble)


def set_size(fraction=1, subplots=(1, 1)) -> tuple[float, float]:
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of the figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = 369 * fraction  # = 129.69 mm
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt  # plus one for left overflow

    text_lines = inches_per_pt * 28
    # Figure height in inches
    fig_height_in = text_lines + fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return fig_width_in, fig_height_in


def multiple_formatter(denominator=2, number=np.pi, latex=r"\pi"):
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def _multiple_formatter(x):
        den = denominator
        num = int(np.rint(den * x / number))
        com = gcd(num, den)
        (num, den) = (int(num / com), int(den / com))
        if den == 1:
            if num == 0:
                return r"$0$"
            if num == 1:
                return r"$%s$" % latex
            elif num == -1:
                return r"$-%s$" % latex
            else:
                return r"$%s%s$" % (num, latex)
        else:
            if num == 1:
                return r"$\frac{%s}{%s}$" % (latex, den)
            elif num == -1:
                return r"$\frac{-%s}{%s}$" % (latex, den)
            else:
                return r"$\frac{%s%s}{%s}$" % (num, latex, den)

    return _multiple_formatter


class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex=r"\pi"):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))
