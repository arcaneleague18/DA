import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = fetch_california_housing()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['Price'] = data.target

X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"\nMAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

results = pd.DataFrame({
    'Actual': y_test.values[:20].round(2),
    'Predicted': y_pred[:20].round(2)
})

print(f"\nPrediction Table:\n{results.to_string(index=False)}")

plt.scatter(y_test, y_pred, alpha=0.4, s=10)

plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    'r',
    label='Perfect Prediction'
)

plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Predictive Analysis - Actual vs Predicted')
plt.legend()

plt.show()
