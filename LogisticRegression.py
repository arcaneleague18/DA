import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

data = load_breast_cancer()
X = data.data[:, [0]]
y = 1 - data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

X_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_prob = model.predict_proba(X_range)[:, 1]

plt.scatter(X_test, y_test, alpha=0.5, c=y_test, cmap='bwr', s=30, label='Data Points')
plt.plot(X_range, y_prob, color='green', linewidth=2, label='Sigmoid Curve')
plt.xlabel('Mean Radius')
plt.ylabel('Probability')
plt.title('Logistic Regression - Breast Cancer (S-Curve)')
plt.legend()
plt.show()
