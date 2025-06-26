import pandas as pd
import numpy as np
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, mean_absolute_error
import matplotlib.pyplot as plt

df_data = pd.read_csv('./data/ml_dataset_2019_2025_20250521_480_train_60_test.csv')
df_data = df_data[df_data['J_Emb_0'].notna()]
df_data['top_three'] = df_data['Placing'].apply(lambda x: 1 if x <= 3 else 0)

cols = df_data.columns.tolist()
embedding_cols = [col for col in cols if col.startswith('H_Emb_') or col.startswith('J_Emb_')]
trailing_cols = [col for col in cols if 'Placing_TR' in col]
race_cols = [
    'Date', 'RaceNumber', 'Horse', 'Placing', 'top_three',
    'Win Odds', 'Dr.', 'Act. Wt.', 'Declar. Horse Wt.', 'DistanceMeter'
]

features = race_cols + trailing_cols + embedding_cols
df_encoded = df_data[features].copy()

df_train = df_encoded[df_encoded['Date'] < '2025-03-22'].copy()
df_test = df_encoded[df_encoded['Date'] >= '2025-03-22'].copy()

X_train = df_train.drop(columns=['top_three', 'Placing', 'Date', 'RaceNumber', 'Horse'], errors='ignore')
y_train = df_train['top_three']

X_train = X_train.select_dtypes(include=[np.number])

params = {
    'objective': 'binary:logistic',
    'max_depth': 3,
    'min_child_weight': 1,
    'gamma': 1,
    'subsample': 1,
    'colsample_bytree': 0.5,
    'learning_rate': 0.01,
    'n_estimators': 100,
    'use_label_encoder': False,
    'eval_metric': 'logloss'
}

model = XGBClassifier(**params)

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, zero_division=0),
    'recall': make_scorer(recall_score, zero_division=0),
    'mae': make_scorer(mean_absolute_error)
}

cv_results = cross_validate(model, X_train, y_train, cv=5, scoring=scoring)

print("\nXGBoost 5-Fold Cross-Validation Results:")
for metric in ['accuracy', 'precision', 'recall', 'mae']:
    mean = cv_results[f'test_{metric}'].mean()
    std = cv_results[f'test_{metric}'].std()
    print(f"{metric.capitalize()}: {mean:.3f} ± {std:.3f}")

model.fit(X_train, y_train)
X_test = df_test.drop(columns=['top_three', 'Placing', 'Date', 'RaceNumber', 'Horse'], errors='ignore')
X_test = X_test.select_dtypes(include=[np.number])
y_test = df_test['top_three']
y_probs = model.predict_proba(X_test)[:, 1]
preds = (y_probs > 0.5).astype(int)

print("\nFinal XGBoost Model:")
print(f"mean_absolute_error: {mean_absolute_error(y_test, preds):.2f}")
print(f"Accuracy: {accuracy_score(y_test, preds):.2f}")
print(f"Precision: {precision_score(y_test, preds, zero_division=0):.2f}")
print(f"Recall: {recall_score(y_test, preds, zero_division=0):.2f}")

plot_importance(model, importance_type='gain', max_num_features=20)
plt.title("Top 20 Feature Importances (Gain)")
plt.show()
