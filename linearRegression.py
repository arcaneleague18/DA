import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = fetch_california_housing()
X = data.data[:, [0]]
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Coefficient: {model.coef_[0]}, Intercept: {model.intercept_}")
print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
print(f"R² Score: {r2_score(y_test, y_pred)}")

plt.scatter(X_test, y_test, alpha=0.4, s=10, label='Data Points')
plt.plot(sorted(X_test), sorted(y_pred), color='red', linewidth=2, label='Regression Line')
plt.xlabel('Median Income')
plt.ylabel('House Value')
plt.title('Linear Regression - California Housing')
plt.legend()
plt.show()
