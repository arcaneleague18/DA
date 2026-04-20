import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = fetch_openml(
    name='weather',
    version=1,
    as_frame=True,
    parser='auto'
).frame

df = df.select_dtypes(include=np.number).dropna()

target = df.columns[-1]

if 'temperature' in df.columns:
    X = df[['temperature']]
else:
    X = df.iloc[:, [0]]

y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

results = pd.DataFrame({
    'Actual': y_test.values[:20].round(2),
    'Predicted': y_pred[:20].round(2)
})

results['Error'] = (results['Actual'] - results['Predicted']).round(2)

print(f"\nWeather Forecast Table:\n{results.to_string(index=False)}")

X_sorted = X.sort_values(by=X.columns[0])
y_sorted_pred = model.predict(X_sorted)

plt.scatter(X_test, y_test, alpha=0.5, s=20, color='green', label='Actual Test Points')
plt.scatter(X_test, y_pred, alpha=0.5, s=20, color='red', marker='x', label='Predicted Points')

plt.plot(X_sorted, y_sorted_pred, color='black', linewidth=2, label='Regression Line')

plt.xlabel(X.columns[0])
plt.ylabel(target)
plt.title('Weather Forecasting - Linear Regression')

plt.legend()
plt.show()
