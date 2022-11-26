import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.preprocessing import StandardScaler

# Read the training data
print("Reading input...")
train_df = pd.read_csv("train.csv", index_col=None)
Y = train_df["TenYearCHD"]
X = train_df.drop("TenYearCHD", axis=1)

print("Scaling...")
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

print("Fitting...")
logreg = LogisticRegression()
logreg.fit(X_scaled, Y)
y_pred = logreg.predict(X_scaled)

# Get the accuracy on the training data
train_accuracy = logreg.score(X_scaled, Y)
print(f"Training accuracy = {train_accuracy}")

# Write out the scaler and logisticregression objects into a pickle file
pickle_path = "classifier.pkl"
print(f"Writing scaling and logistic regression model to {pickle_path}...")
output = open(pickle_path, "wb")
pickle.dump((scaler,logreg), output)
output.close()
