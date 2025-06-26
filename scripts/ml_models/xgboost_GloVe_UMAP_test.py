from datetime import datetime, timedelta
import numpy as np
import xgboost as xgb
from xgboost import plot_importance
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import sys
import os

from xgboost.sklearn import XGBClassifier
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.dirname(current_dir)
sys.path.append(scripts_dir)
import query_repository as qr
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


df_data = pd.read_csv('./data/ml_dataset_2019_2025_390_train_30_test_all.csv')
df_data['top_three'] = df_data['Placing'].apply(lambda x: 1 if x <= 3 else 0)
# Exclude rows where jockey embedding data is uknown
df_data= df_data[df_data['J_Emb_0'].notna()]
# df_data= df_data[df_data['H_Emb_0'].notna()]

effective_date_list = df_data['Effective Date'].unique().tolist()
cols = df_data.columns.tolist()


h_emb_cols = [col for col in cols if col[0:6] in ['J_Emb_', 'J_UMAP'] and 'rank_' not in col]
j_emb_cols = [col for col in cols if col[0:6] in ['H_Emb_', 'H_UMAP'] and 'rank_' not in col]
# h_emb_cols = [col for col in cols if col[0:6] in ['J_Emb_', 'J_UMAP']]
# j_emb_cols = [col for col in cols if col[0:6] in ['H_Emb_', 'H_UMAP']]


test_dates_start = []
test_dates_end = []
for effective_date in effective_date_list:

    keep_cols = ['Date', 'RaceNumber', 'Horse', 'Placing', 'top_three', 
                #  'Win Odds'
                ]

    numeric_cols = ['Placing_TR1','Placing_TR2','Placing_TR3','Placing_TR4',
                    'Placing_TR5','Placing_TR6','Placing_TR7','Placing_TR8',
                    'Placing_TR9','Placing_TR10']

    cols = df_data.columns.tolist()

    embedding_cols = []
    embedding_cols = h_emb_cols + j_emb_cols
    umap_cols = [col for col in cols if col[0:6] in ['H_UMAP', 'J_UMAP']]
    glove_cols = [col for col in cols if col[0:6] in ['H_Emb_', 'J_Emb']]

    best_iteration = [0, 0, 0, 0, 0]
    best_features = []
    for i in range(0, 1):
        #  embedding_cols_rand + 
        features = set(
                         keep_cols
                       + numeric_cols                    
                       + glove_cols
                       + umap_cols
                    )
        features = list(features)

        df_encoded = df_data[df_data['Effective Date'] == effective_date][features].copy()
        df_train = df_encoded[df_encoded['Date'] < effective_date].copy()
        effective_date_end = (datetime.strptime(effective_date, '%Y-%m-%d') + timedelta(days=30)).strftime('%Y-%m-%d')
        df_test = df_encoded[(df_encoded['Date'] >= effective_date) & (df_encoded['Date'] < effective_date_end)].copy()

        test_dates_start.append(effective_date)
        test_dates_end.append(effective_date_end)

        # print(len(df_train), len(df_test), len(df_encoded), len(df_train) + len(df_test))
        X_train = df_train.drop(columns=['top_three'])
        X_test = df_test.drop(columns=['top_three'])
        # X_test = df_test.drop(columns=['top_three',  '_merge'])
        y_train = df_train['top_three']
        y_test = df_test['top_three']

        X_train_drop = X_train.drop(columns=['Placing', 'Date', 'RaceNumber', 'Horse'], errors='ignore')
        X_test_drop = X_test.drop(columns=['Placing', 'Date', 'RaceNumber', 'Horse'], errors='ignore')

        dtrain = xgb.DMatrix(X_train_drop, label=y_train)
        dtest = xgb.DMatrix(X_test_drop, label=y_test)

        #     'eval_metric': 'mlogloss',
        params = {

            'objective': 'binary:logistic',
            'max_depth': 5,
            'min_child_weight': 1,
            'gamma': .55,
            'subsample': 0.85,
            'colsample_bytree': 1,
            'learning_rate':0.01, 
            'n_estimators':1000,
            # 'early_stopping_rounds': 100,
        }


        # params = {
        #     'colsample_bytree': 0.8893244572081387,
        #     'gamma': 1.0,
        #     'learning_rate': 0.29999999999999993,
        #     'max_depth': 3,
        #     'min_child_weight': 10,
        #     'n_estimators': 100,
        #     'subsample': 0.5
        #}


        model = XGBClassifier(**params)
        model.fit(X_train_drop, y_train)

        #preds = model.predict(dtest)
        y_probs = model.predict_proba(X_test_drop)[:, 1]
        preds = (y_probs > 0.5).astype(int)

        df_preds = pd.DataFrame(preds, columns=['Predicted Placing'])
        df_preds['top_three'] = y_test.values

        df_merged = pd.concat([X_test[['Horse', 'Date', 'RaceNumber', 'Placing']], df_preds], axis=1)

        mean_absolute_error = 0.0
        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)

        # print(str(accuracy) + "," + str(precision) + "," + str(recall))
        # if precision > best_iteration[1]:
        #     print(str(accuracy) + "," + str(precision) + "," + str(recall))
        #     best_iteration = [accuracy, precision, recall, accuracy, mean_absolute_error]
        #     best_features = features
        #     print(f'New best iteration: {i}')
        #     print(f'Precision: {precision:.2f}, Recall: {recall:.2f}, Accuracy: {accuracy:.2f}, MAE: {mean_absolute_error:.2f}')
        #     print(f'Features: {features}')


        # plot_importance(model, importance_type='gain')
        # plt.show()


        # Print Feature Importance for Feature Importance Plot and Analysis
        # importance_scores = model.get_booster().get_score(importance_type='gain')
        # for feature, score in importance_scores.items():            
        #     print(feature + "," + str(score))

# for start, end in zip(test_dates_start, test_dates_end):
#     print(str(start) + "," + str(end))
