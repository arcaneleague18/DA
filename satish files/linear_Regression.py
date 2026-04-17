import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([[1, 2],
              [2, 3],
              [3, 4],
              [4, 5]])

y = np.array([3, 5, 7, 9])

model = LinearRegression()

model.fit(X, y)

pred = model.predict(X)

print("Predictions:", pred)

plt.scatter(X[:,0], y)
plt.plot(X[:,0], pred)

plt.title("Linear Regression")
plt.xlabel("X")
plt.ylabel("Y")

plt.show()