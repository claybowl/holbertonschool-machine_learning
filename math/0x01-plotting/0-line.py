#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

# Plot y as a line graph
plt.plot(y, '-r') # '-r' for solid red line

# Set x-axis range from 0 to 10
plt.xlim(0, 10)

# Display the plot
plt.show()
