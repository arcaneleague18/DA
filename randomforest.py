# RANDOM FOREST
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd, matplotlib.pyplot as plt

data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=2, max_depth=3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Accuracy:  {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
print(f"Recall:    {recall_score(y_test, y_pred, average='weighted')}")
print(f"F1 Score:  {f1_score(y_test, y_pred, average='weighted')}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

fig, axes = plt.subplots(1, 2, figsize=(40, 15))
for i, tree in enumerate(model.estimators_):
    plot_tree(tree, feature_names=data.feature_names, class_names=data.target_names,
              filled=True, rounded=True, ax=axes[i], fontsize=20)
    axes[i].set_title(f'Tree {i+1}')
plt.tight_layout()
plt.show()
