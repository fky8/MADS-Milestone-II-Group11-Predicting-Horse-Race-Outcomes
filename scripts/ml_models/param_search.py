import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_absolute_error
from skopt import BayesSearchCV
from skopt.space import Real, Integer

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
X_test = df_test.drop(columns=['top_three', 'Placing', 'Date', 'RaceNumber', 'Horse'], errors='ignore')
y_train = df_train['top_three']
y_test = df_test['top_three']

X_train = X_train.select_dtypes(include=[np.number])
X_test = X_test.select_dtypes(include=[np.number])

param_space = {
    'max_depth': Integer(3, 10),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'n_estimators': Integer(100, 1000),
    'gamma': Real(0, 1),
    'subsample': Real(0.5, 1.0),
    'colsample_bytree': Real(0.5, 1.0),
    'min_child_weight': Integer(1, 10)
}

xgb_model = XGBClassifier(
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric='logloss',
    verbosity=0
)

opt = BayesSearchCV(
    xgb_model,
    search_spaces=param_space,
    n_iter=25,
    scoring='precision',
    cv=3,
    n_jobs=-1,
    random_state=42,
    verbose=0
)

opt.fit(X_train, y_train)

print("\nBest parameters:")
print(opt.best_params_)

best_model = opt.best_estimator_
y_probs = best_model.predict_proba(X_test)[:, 1]
preds = (y_probs > 0.5).astype(int)

print("\nBest Model:")
print(f"mean_absolute_error: {mean_absolute_error(y_test, preds):.2f}")
print(f"Accuracy: {accuracy_score(y_test, preds):.2f}")
print(f"Precision: {precision_score(y_test, preds, zero_division=0):.2f}")
print(f"Recall: {recall_score(y_test, preds, zero_division=0):.2f}")

plot_importance(best_model, importance_type='gain', max_num_features=20)
plt.title("Top 20 Feature Importances")
plt.show()