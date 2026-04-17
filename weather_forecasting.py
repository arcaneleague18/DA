from sklearn.linear_model import LinearRegression
import numpy as np

days = np.array([[1],[2],[3],[4],[5]])
temp = np.array([30,32,35,37,40])

model = LinearRegression()
model.fit(days,temp)

forecast = model.predict([[6],[7]])

print("Forecast Temp:", forecast)
print("Score:", model.score(days,temp))