from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1],[2],[3],[4],[5]])
y = np.array([2,4,6,8,10])

model = LinearRegression()
model.fit(X,y)

pred = model.predict([[6]])

print("Coef:", model.coef_)
print("Intercept:", model.intercept_)
print("Prediction:", pred)