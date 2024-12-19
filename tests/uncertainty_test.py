from uncertainties import ufloat
import numpy as np

a = ufloat(1, 0.1, tag="a")
b = ufloat(1.2, 0.2, tag="b")
a2 = a * 2
b2 = a * 3 - b
c = a**2 + b * a - b**2

# (dc/da) = 2a +b
# (dc/db) = -b
dcda = 2 * a.nominal_value + b.nominal_value
dcdb = a.nominal_value - 2 * b.nominal_value

sc = np.sqrt(dcda**2 * a.std_dev**2 + dcdb**2 * b.std_dev**2)
c = a2 * b2
print(f"{c=}")
print(f"{c.std_dev=}")
print(f"{sc=}")
sc_from_components = 0
for var, value in c.error_components().items():
    print(var.tag, value)
    sc_from_components += value**2
sc_from_components = np.sqrt(sc_from_components)
print(sc_from_components)

syst_error = np.sqrt(
    sum(  # Error from *all* systematic errors
        error**2 for (var, error) in c.error_components().items() if var.tag
    )
)
print(syst_error)
