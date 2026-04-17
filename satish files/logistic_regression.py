import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

X = np.array([[1, 2],
              [2, 3],
              [3, 4],
              [4, 5]])

y = np.array([0, 0, 1, 1])

model = LogisticRegression()

model.fit(X, y)

pred = model.predict(X)

print("Predictions:", pred)

plt.scatter(X[:,0], y)
plt.scatter(X[:,0], pred)

plt.title("Logistic Regression")
plt.xlabel("X")
plt.ylabel("Class")

plt.show()