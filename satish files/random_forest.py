import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

X = np.array([[1, 2],
              [2, 3],
              [3, 4],
              [4, 5]])

y = np.array([0, 1, 0, 1])

model = RandomForestClassifier(n_estimators=10)

model.fit(X, y)

pred = model.predict(X)

print("Predictions:", pred)

plt.scatter(X[:,0], y)
plt.scatter(X[:,0], pred)

plt.title("Random Forest")
plt.show()