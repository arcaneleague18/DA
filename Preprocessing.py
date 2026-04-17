import pandas as pd
import numpy as np

# Create data
df = pd.DataFrame({
    'A':[1,2,np.nan,4,100],
    'B':[5,np.nan,7,8,9]
})

# Fill missing values
df = df.fillna(df.mean())

# Remove noise (replace large values)
df['A'] = df['A'].apply(lambda x: df['A'].mean() if x > 50 else x)

# Remove redundancy
df['C'] = df['A']
df = df.drop(columns=['C'])

# Display result
print(df)