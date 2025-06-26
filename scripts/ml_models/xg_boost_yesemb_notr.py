import pandas as pd
import numpy as np
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_absolute_error
import matplotlib.pyplot as plt

df_data = pd.read_csv('./data/ml_dataset_2019_2025_20250521_480_train_60_test.csv')
df_data = df_data[df_data['J_Emb_0'].notna()]
df_data['top_three'] = df_data['Placing'].apply(lambda x: 1 if x <= 3 else 0)

cols = df_data.columns.tolist()
embedding_cols = [col for col in cols if col.startswith('H_Emb_') or col.startswith('J_Emb_')]
race_cols = [
    'Date', 'RaceNumber', 'Horse', 'Placing', 'top_three',
    'Win Odds', 'Dr.', 'Act. Wt.', 'Declar. Horse Wt.', 'DistanceMeter'
]

features = race_cols + embedding_cols
df_encoded = df_data[features].copy()

df_train = df_encoded[df_encoded['Date'] < '2025-03-22'].copy()
df_test = df_encoded[df_encoded['Date'] >= '2025-03-22'].copy()

X_train = df_train.drop(columns=['top_three', 'Placing', 'Date', 'RaceNumber', 'Horse'], errors='ignore')
X_test = df_test.drop(columns=['top_three', 'Placing', 'Date', 'RaceNumber', 'Horse'], errors='ignore')
y_train = df_train['top_three']
y_test = df_test['top_three']

X_train = X_train.select_dtypes(include=[np.number])
X_test = X_test.select_dtypes(include=[np.number])

params = {
    'objective': 'binary:logistic',
    'max_depth': 4,
    'min_child_weight': 1,
    'gamma': 0.55,
    'subsample': 0.85,
    'colsample_bytree': 1,
    'learning_rate': 0.01,
    'n_estimators': 1000,
    'use_label_encoder': False,
    'eval_metric': 'logloss'
}

model = XGBClassifier(**params)
model.fit(X_train, y_train)
y_probs = model.predict_proba(X_test)[:, 1]
preds = (y_probs > 0.5).astype(int)

print("\nFinal XGBoost Model (Embeddings + Race Features Only):")
print(f"mean_absolute_error: {mean_absolute_error(y_test, preds):.2f}")
print(f"Accuracy: {accuracy_score(y_test, preds):.2f}")
print(f"Precision: {precision_score(y_test, preds, zero_division=0):.2f}")
print(f"Recall: {recall_score(y_test, preds, zero_division=0):.2f}")

plot_importance(model, importance_type='gain', max_num_features=20)
plt.title("Top 20 Feature Importances (Gain)")
plt.show()