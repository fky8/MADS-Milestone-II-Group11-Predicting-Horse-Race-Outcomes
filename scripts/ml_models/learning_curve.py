import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_absolute_error
import matplotlib.pyplot as plt

base_path = './data'
prefix = 'ml_dataset_2019_2025_20250521_'
train_days = list(range(60, 541, 30))  
test_date_cutoff = '2025-03-22'

results = {
    'train_days': [],
    'accuracy': [],
    'precision': [],
    'recall': [],
    'mae': []
}

for days in train_days:
    fname = f'{prefix}{days}_train_60_test.csv'
    path = os.path.join(base_path, fname)
    print(f"Processing: {fname}")

    df = pd.read_csv(path)
    df = df[df['J_Emb_0'].notna()]
    df['top_three'] = df['Placing'].apply(lambda x: 1 if x <= 3 else 0)

    cols = df.columns.tolist()
    embedding_cols = [col for col in cols if col.startswith('H_Emb_') or col.startswith('J_Emb_')]
    trailing_cols = [col for col in cols if 'Placing_TR' in col]
    race_cols = [
        'Date', 'RaceNumber', 'Horse', 'Placing', 'top_three',
        'Win Odds', 'Dr.', 'Act. Wt.', 'Declar. Horse Wt.', 'DistanceMeter'
    ]

    features = race_cols + trailing_cols + embedding_cols
    df_encoded = df[features].copy()

    df_train = df_encoded[df_encoded['Date'] < test_date_cutoff].copy()
    df_test = df_encoded[df_encoded['Date'] >= test_date_cutoff].copy()

    X_train = df_train.drop(columns=['top_three', 'Placing', 'Date', 'RaceNumber', 'Horse'], errors='ignore')
    X_test = df_test.drop(columns=['top_three', 'Placing', 'Date', 'RaceNumber', 'Horse'], errors='ignore')
    y_train = df_train['top_three']
    y_test = df_test['top_three']

    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])

    model = XGBClassifier(
        objective='binary:logistic',
        max_depth=3,
        min_child_weight=1,
        gamma=1,
        subsample=1,
        colsample_bytree=0.5,
        learning_rate=0.01,
        n_estimators=100,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)
    y_probs = model.predict_proba(X_test)[:, 1]
    preds = (y_probs > 0.5).astype(int)

    results['train_days'].append(days)
    results['accuracy'].append(accuracy_score(y_test, preds))
    results['precision'].append(precision_score(y_test, preds, zero_division=0))
    results['recall'].append(recall_score(y_test, preds, zero_division=0))
    results['mae'].append(mean_absolute_error(y_test, preds))

results_df = pd.DataFrame(results)

plt.figure(figsize=(12, 6))
plt.plot(results_df['train_days'], results_df['accuracy'], label='Accuracy')
plt.plot(results_df['train_days'], results_df['precision'], label='Precision')
plt.plot(results_df['train_days'], results_df['recall'], label='Recall')
plt.xlabel("Training Days")
plt.ylabel("Score")
plt.title("Learning Curve: Model Performance vs. Training Set Size")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
