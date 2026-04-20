import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset('healthexp')

print("Shape:", df.shape)

print("\nFirst 10 Rows:\n", df.head(10))

print("\nData Types:\n", df.dtypes)

print("\nBasic Statistics:\n", df.describe())

print("\nMedian:\n", df.median(numeric_only=True))

print("\nMode:\n", df.mode(numeric_only=True).iloc[0])

print("\nVariance:\n", df.var(numeric_only=True))

print("\nStandard Deviation:\n", df.std(numeric_only=True))

print("\nSkewness:\n", df.skew(numeric_only=True))

print("\nKurtosis:\n", df.kurtosis(numeric_only=True))

print("\nCorrelation:\n", df.corr(numeric_only=True))

print("\nMissing Values:\n", df.isnull().sum())

print("\nUnique Countries:", df['Country'].nunique())

print(
    "\nCountry-wise Avg Spending:\n",
    df.groupby('Country')['Spending_USD'].mean().round(2)
)

print(
    "\nCountry-wise Avg Life Expectancy:\n",
    df.groupby('Country')['Life_Expectancy'].mean().round(2)
)

print("\nYear Range:", df['Year'].min(), "-", df['Year'].max())
