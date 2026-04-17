from sklearn.linear_model import LogisticRegression
import numpy as np

X = np.array([[1],[2],[3],[4],[5],[6]])
y = np.array([0,0,0,1,1,1])

model = LogisticRegression()
model.fit(X,y)

pred = model.predict([[2.5],[5]])

print("Classes:", pred)
print("Score:", model.score(X,y))