import pandas as pd
from sklearn.datasets import load_diabetes

# Load the diabetes dataset from sklearn
data = load_diabetes(as_frame=True)
df = data.frame.copy()

# Convert the continuous target into binary: 1 = high risk, 0 = low risk
df['target'] = (df['target'] > df['target'].mean()).astype(int)

# Save it to CSV format
df.to_csv('data/diabetes_sample_data.csv', index=False)
