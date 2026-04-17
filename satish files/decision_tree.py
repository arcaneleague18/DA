import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

X = np.array([[1, 2],
              [2, 3],
              [3, 4],
              [4, 5]])

y = np.array([0, 0, 1, 1])

model = DecisionTreeClassifier()

model.fit(X, y)

pred = model.predict(X)

print("Predictions:", pred)

plt.scatter(X[:,0], y)
plt.scatter(X[:,0], pred)

plt.title("Decision Tree Classification")
plt.show()

plt.figure(figsize=(8,6))

plot_tree(model, feature_names=["Feature1", "Feature2"],
          class_names=["0", "1"], filled=True)

plt.title("Decision Tree Structure")
plt.show()