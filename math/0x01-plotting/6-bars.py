#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

people = ['Farrah', 'Fred', 'Felicia']
fruits = ['apples', 'bananas', 'oranges', 'peaches']
colors = ['red', 'yellow', 'orange', 'peach']

# Set the x axis as people
x = np.arange(len(people))

# Plot the stacked bar graph
plt.bar(x, fruit[0], color=colors[0], width=0.5)
for i in range(1, fruit.shape[0]):
    plt.bar(x, fruit[i], bottom=fruit[:i].sum(axis=0), color=colors[i], width=0.5)

# Set the x-axis labels as people
plt.xticks(x, people)

# Label the y-axis as "Quantity of Fruit"
plt.ylabel('Quantity of Fruit')

# Set the title as "Number of Fruit per Person"
plt.title("Number of Fruit per Person")

# Set the y-axis range from 0 to 80 with ticks every 10 units
plt.ylim(0, 80)
plt.yticks(np.arange(0, 81, 10))

# Create a legend
plt.legend(fruits, bbox_to_anchor=(1, 1), loc='upper left')

# Show the plot
plt.show()
