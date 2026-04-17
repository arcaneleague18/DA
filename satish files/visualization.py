import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1, 2],
              [2, 4],
              [3, 6],
              [4, 8]])

x = X[:, 0]
y = X[:, 1]

plt.figure()
plt.plot(x, y)
plt.title("Line Plot")
plt.show()

plt.figure()
plt.bar(x, y)
plt.title("Bar Plot")
plt.show()

plt.figure()
plt.scatter(x, y)
plt.title("Scatter Plot")
plt.show()