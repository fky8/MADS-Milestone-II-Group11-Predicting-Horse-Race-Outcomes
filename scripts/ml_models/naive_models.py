import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_absolute_error

df_data = pd.read_csv('./data/ml_dataset_2019_2025_20250521_480_train_60_test.csv')
df_data['top_three'] = df_data['Placing'].apply(lambda x: 1 if x <= 3 else 0)

df_train = df_data[df_data['Date'] < '2025-01-01'].copy()
df_test = df_data[df_data['Date'] >= '2025-01-01'].copy()

y_test = df_test['top_three']

preds = [0] * len(df_test)

mae = mean_absolute_error(y_test, preds)
accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, zero_division=0)
recall = recall_score(y_test, preds, zero_division=0)

print("\nDummy Variable:")
print(f"mean_absolute_error: {mae:.2f}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")


df_train = df_data[df_data['Date'] < '2025-01-01'].copy()
df_test = df_data[df_data['Date'] >= '2025-01-01'].copy()

y_test = df_test['top_three']

class_probs = df_train['top_three'].value_counts(normalize=True).sort_index()

np.random.seed(42)
preds = np.random.choice([0, 1], size=len(y_test), p=class_probs.values)

mae = mean_absolute_error(y_test, preds)
accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, zero_division=0)
recall = recall_score(y_test, preds, zero_division=0)

print("\nRandom Choice:")
print(f"mean_absolute_error: {mae:.2f}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")