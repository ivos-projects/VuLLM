from src.utils import random_classifier
from datasets import load_dataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

dataset = load_dataset("oscaraandersson/reveal")
fraction_postive = sum(dataset["train"]["output"]) / len(dataset["train"]["output"])
n = len(dataset["test"]["output"])
weights = {1: fraction_postive, 0: 1 - fraction_postive}
y_pred_weighted = random_classifier(weights, n)
y_pred = random_classifier({1: 0.5, 0: 0.5}, n)
y_true = dataset["test"]["output"]


# Assuming y_true and y_pred are your true and predicted labels, respectively
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)


# Print the results
print("Results for 50/50 classifier:")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}\n\n")

precision = precision_score(y_true, y_pred_weighted)
recall = recall_score(y_true, y_pred_weighted)
f1 = f1_score(y_true, y_pred_weighted)
accuracy = accuracy_score(y_true, y_pred_weighted)


# Print the results
print("Results for weighted classifier:")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")
