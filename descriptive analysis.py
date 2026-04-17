import pandas as pd
# healthcare dataset made by me
df = pd.DataFrame({
    'age':[25,40,60,35,50],
    'bp':[120,140,150,130,145]
})

print("Mean Age:", df['age'].mean())
print("Max BP:", df['bp'].max())
print("Summary:\n", df.describe())

print("Median Age:", df['age'].median())
print("Min BP:", df['bp'].min())