from sklearn.tree import DecisionTreeClassifier
import numpy as np

X = [[1,2],[2,3],[3,4],[4,5]]
y = [0,0,1,1]

model = DecisionTreeClassifier()
model.fit(X,y)

pred = model.predict([[3,3]])

print("Prediction:", pred)
print("Accuracy:", model.score(X,y))