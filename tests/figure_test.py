"""
Creates a test figure with the helpers.figures width and tests the pgf export
"""

import matplotlib.pyplot as plt

from helpers import figures

# Create some sample data
x = range(10)
y = range(10)

# Set the size of the data area in inches
data_width, data_height = figures.set_size()

# Calculate the figure size based on the data area size
# Add some padding for the axis labels and ticks
padding = 1
fig_width = data_width
fig_height = data_height

# Create a figure with the specified size
fig = plt.figure(figsize=figures.set_size())

# Add a subplot with the specified size
ax = fig.add_subplot(111)

# Plot the data
ax.plot(x, y)

# Adjust the subplot parameters to fit the data area
fig.subplots_adjust(left=1 / fig_width, right=1 - 1 / fig_width)

plt.savefig(
    "test.pdf",
    backend="pgf",
)
plt.savefig(
    "test.pgf",
    backend="pgf",
)
