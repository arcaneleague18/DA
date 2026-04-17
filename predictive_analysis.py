from sklearn.linear_model import LinearRegression
import numpy as np

days = np.array([[1],[2],[3],[4],[5]])
sales = np.array([100,150,200,250,300])

model = LinearRegression()
model.fit(days,sales)

future = model.predict([[6],[7]])

print("Future Sales:", future)
print("Coef:", model.coef_)