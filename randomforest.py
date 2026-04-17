from sklearn.ensemble import RandomForestClassifier

X = [[1,2],[2,3],[3,4],[4,5]]
y = [0,0,1,1]

model = RandomForestClassifier(n_estimators=10)
model.fit(X,y)

pred = model.predict([[3,3]])

print("Prediction:", pred)
print("Score:", model.score(X,y))