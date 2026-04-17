import pandas as pd
import numpy as np

df = pd.DataFrame({
    'x':[1,2,3,4,5],
    'y':[2,4,6,8,10],
    'z':[5,3,6,2,1]
})

corr = df.corr()
print("Correlation Matrix:\n", corr)

print("x vs y:", corr['x']['y'])
print("x vs z:", corr['x']['z'])