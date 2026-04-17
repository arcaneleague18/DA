import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([[1, 100],
              [2, 150],
              [3, 200],
              [4, 250]])

y = np.array([200, 300, 400, 500])

model = LinearRegression()

model.fit(X, y)

future = np.array([[5, 300]])

pred = model.predict(future)

print("Future Prediction:", pred)

plt.plot(y)
plt.title("Sales Trend")
plt.show()