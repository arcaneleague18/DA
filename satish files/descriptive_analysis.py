import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X = np.array([[25, 120],
              [35, 130],
              [45, 140],
              [55, 150]])

df = pd.DataFrame(X, columns=['Age', 'BP'])

print(df.describe())

plt.hist(df['Age'])
plt.title("Age Distribution")
plt.show()

plt.plot(df['BP'])
plt.title("BP Trend")
plt.show()