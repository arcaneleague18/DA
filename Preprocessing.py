import numpy as np
import pandas as pd
import seaborn as sns

df = sns.load_dataset('titanic')

print("Original Shape:", df.shape)
print("Missing Values:\n", df.isnull().sum())

df['age'] = df['age'].fillna(df['age'].median())

df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df['deck'] = df['deck'].fillna(df['deck'].mode()[0])
df['embark_town'] = df['embark_town'].fillna(df['embark_town'].mode()[0])

print("\nAfter Imputation:\n", df.isnull().sum())

for col in df.select_dtypes(include=np.number).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    lb = Q1 - 1.5 * (Q3 - Q1)
    ub = Q3 + 1.5 * (Q3 - Q1)

    out = ((df[col] < lb) | (df[col] > ub)).sum()

    if out > 0:
        print(f"{col}: {out} outliers clipped to [{lb:.2f}, {ub:.2f}]")

    df[col] = df[col].clip(lb, ub)

print(f"\nDuplicate Rows Found: {df.duplicated().sum()}")

df = df.drop_duplicates()

df = df.drop(columns=['alive', 'embark_town', 'who'])

print("Dropped Redundant Columns: alive, embark_town, who")

print(f"\nFinal Shape: {df.shape}")
print(df.describe())
