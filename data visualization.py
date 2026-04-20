import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes

data = load_diabetes()

df = pd.DataFrame({
    'BMI': data.data[:,2],
    'BP': data.data[:,3],
    "Target": data.target
})

df_small = df.head(5).reset_index()
df_melt = df_small.melt(id_vars='index', value_vars=['BMI', "BP"])

sns.barplot(x = 'index', y='value', hue='variable', data=df_melt)
plt.show()

sns.barplot(x = 'index', y= 'value', hue='variable', data=df_melt, orient='h')
plt.show()

sns.scatterplot(x = 'BMI', y = 'Target', data=df)
plt.show()

sns.lineplot(x = 'BMI', y='Target', data = df)
plt.show()
