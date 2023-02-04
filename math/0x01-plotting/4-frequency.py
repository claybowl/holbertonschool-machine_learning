#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# Plot the histogram
plt.hist(student_grades, bins=np.arange(0, 110, 10), edgecolor='black')

# Label the x-axis as "Grades"
plt.xlabel('Grades')

# Label the y-axis as "Number of Students"
plt.ylabel('Number of Students')

# Set the title as "Project A"
plt.title("Project A")

# Display the plot
plt.show()
