import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = fetch_openml(name='weather', version=1, as_frame=True).frame
df = df.select_dtypes(include=np.number).dropna()

target = df.columns[-1]
X = df[['temperature']] if 'temperature' in df.columns else df.iloc[:, [0]]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)

y_pred = model.predict(X_test)

print(pd.DataFrame({'Actual': y_test.values[:10], 'Predicted': y_pred[:10]}).round(2))

X_sorted = X.sort_values(by=X.columns[0])
plt.scatter(X_test, y_test, color='green', s=20, label='Actual')
plt.scatter(X_test, y_pred, color='red', s=20, marker='x', label='Predicted')
plt.plot(X_sorted, model.predict(X_sorted), color='black', label='Regression Line')

plt.xlabel(X.columns[0])
plt.ylabel(target)
plt.title('Linear Regression')
plt.legend()
plt.show()
