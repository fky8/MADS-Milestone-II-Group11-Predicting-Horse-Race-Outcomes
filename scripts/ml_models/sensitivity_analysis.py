import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from skopt import BayesSearchCV
from skopt.plots import plot_objective
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

search_space = {
    'max_depth': Integer(3, 8),
    'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
    'min_child_weight': Integer(1, 10),
    'subsample': Real(0.6, 1.0),
    'colsample_bytree': Real(0.6, 1.0),
    'gamma': Real(0, 1.0),
    'n_estimators': Integer(200, 1000)
}

model = XGBClassifier(
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric='logloss'
)

opt = BayesSearchCV(
    estimator=model,
    search_spaces=search_space,
    scoring='precision',
    cv=3,
    n_iter=20,
    random_state=42,
    verbose=1,
    n_jobs=-1
)

opt.fit(X_train, y_train)

results_df = pd.DataFrame(opt.cv_results_)

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
param_names = list(search_space.keys())

for i, param in enumerate(param_names):
    axes[i].scatter(results_df['param_' + param], results_df['mean_test_score'], alpha=0.7)
    axes[i].set_title(f"Precision vs {param}")
    axes[i].set_xlabel(param)
    axes[i].set_ylabel("Precision")

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle("Sensitivity Analysis: Precision vs Hyperparameters", fontsize=16, y=1.02)
plt.show()

print("Best Precision:", opt.best_score_)
print("Best Hyperparameters:", opt.best_params_)