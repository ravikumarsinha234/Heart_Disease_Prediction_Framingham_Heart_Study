import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    recall_score,
    precision_score,
    accuracy_score,
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle
import logging

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
h = logging.StreamHandler()
f = logging.Formatter("%(levelname)s@%(asctime)s: %(message)s", datefmt="%H:%M:%S")
h.setFormatter(f)
logger.addHandler(h)

# Read in test data
test_df = pd.read_csv("test.csv", index_col=None)
Y = test_df["TenYearCHD"]
X = test_df.drop("TenYearCHD", axis=1)

# Read in model
scaler,pickled_model = pickle.load(open("classifier.pkl", "rb"))

# Scale X
#scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# Check accuracy on test data
test_accuracy = pickled_model.score(X_scaled, Y)
print(f"Test Accuracy = {test_accuracy}")

# Show confusion matrix
y_pred = pickled_model.predict(X_scaled)
cm = confusion_matrix(Y, y_pred)
print(f"Confusion matrix = \n{cm}")

# Try a bunch of thresholds
threshold = 0.0
best_f1 = -1.0
thresholds = []
recall_scores = []
precision_scores = []
f1_scores = []
pred_proba = pickled_model.predict_proba(X_scaled)[:, 1]
while threshold <= 1.0:
    thresholds.append(threshold)
    thres_pred_res = [1 if unit_prob > threshold else 0 for unit_prob in pred_proba]
    recall = recall_score(Y, thres_pred_res)
    precision = (
        0 if threshold == 0.0 else precision_score(Y, thres_pred_res, zero_division=1)
    )
    accuracy = accuracy_score(Y, thres_pred_res)
    recall_scores.append(recall)
    precision_scores.append(precision)
    f1 = 2 * (precision * recall) / (precision + recall)
    f1_scores.append(f1)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
    logger.info(
        f"Threshold={threshold:.3f} Accuracy={accuracy:.3f} Recall={recall:.2f} Precision={precision:.2f} F1 = {f1:.3f}"
    )
    threshold += 0.02

# Make a confusion matrix for the best threshold

thres_pred_res = [1 if unit_prob > best_threshold else 0 for unit_prob in pred_proba]
cm_best_threshold = confusion_matrix(Y, thres_pred_res)
print(f"Confusion matrix for best threshold= \n{cm_best_threshold}")

# Plot recall, precision, and F1 vs Threshold
fig, ax = plt.subplots()
ax.plot(thresholds, recall_scores, "b", label="Recall")
ax.plot(thresholds, precision_scores, "g", label="Precision")
ax.plot(thresholds, f1_scores, "r", label="F1")
ax.vlines(best_threshold, 0, 1, "r", linewidth=0.5, linestyle="dashed")
ax.set_xlabel("Threshold")
ax.legend()
fig.savefig("threshold.png")
