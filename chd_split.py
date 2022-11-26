import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("framingham.csv", index_col=False)
print(f"Read {df.shape[0]} rows")

# Drop rows with missing values
df.dropna(inplace=True)

print(f"Using {df.shape[0]} rows")

# Split into training and testing dataframes
train_data, test_data = train_test_split(df, test_size=0.20, shuffle=True)

# Write out each as a CSV
train_data.to_csv("train.csv", index=False)
test_data.to_csv("test.csv", index=False)
