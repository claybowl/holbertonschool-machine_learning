#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

# Plot y1 as a dashed red line
plt.plot(x, y1, '--r', label='C-14')

# Plot y2 as a solid green line
plt.plot(x, y2, '-g', label='Ra-226')

# Label the x-axis as "Time (years)"
plt.xlabel('Time (years)')

# Label the y-axis as "Fraction Remaining"
plt.ylabel('Fraction Remaining')

# Set the title as "Exponential Decay of Radioactive Elements"
plt.title("Exponential Decay of Radioactive Elements")

# Set the x-axis range from 0 to 20,000
plt.xlim(0, 20000)

# Set the y-axis range from 0 to 1
plt.ylim(0, 1)

# Add a legend
plt.legend(loc='upper right')

# Display the plot
plt.show()
