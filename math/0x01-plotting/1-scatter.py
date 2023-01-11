#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x, y = np.random.multivariate_normal(mean, cov, 2000).T
y += 180

# Plot the data as a scatter plot
plt.scatter(x, y, c='magenta', marker='o')

# Label the x-axis as "Height (in)"
plt.xlabel('Height (in)')

# Label the y-axis as "Weight (lbs)"
plt.ylabel('Weight (lbs)')

# Set the title as "Men's Height vs Weight"
plt.title("Men's Height vs Weight")

# Display the plot
plt.show()
