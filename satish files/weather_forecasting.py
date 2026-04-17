import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([[1, 30],
              [2, 32],
              [3, 34],
              [4, 36]])

y = np.array([30, 32, 34, 36])

model = LinearRegression()

model.fit(X, y)

future = np.array([[5, 38]])

pred = model.predict(future)

print("Predicted Temp:", pred)

plt.plot(y)
plt.title("Temperature Trend")
plt.show()