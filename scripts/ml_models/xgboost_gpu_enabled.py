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

df_data = pd.read_csv('./data/ml_dataset_2019_2025_20250521_480_train_60_test.csv')
cols = df_data.columns.tolist()


h_emb_cols = [col for col in cols if col[0:6] in ['J_Emb_']]
j_emb_cols = [col for col in cols if col[0:6] in ['H_Emb_']]

h_cols = ['Date', 'RaceNumber'] + h_emb_cols
j_cols = ['Date', 'RaceNumber'] + j_emb_cols

df_jockey = df_data[j_cols].copy()
df_horse = df_data[h_cols].copy()

# Append Average opponent embeddings
df_jockey_grouped = df_jockey.groupby(['Date', 'RaceNumber']).mean().reset_index()
for col in j_emb_cols:
    df_jockey_grouped.rename(columns={col: col + '_mean'}, inplace=True)

df_horse_grouped = df_horse.groupby(['Date', 'RaceNumber']).mean().reset_index()
for col in h_emb_cols:
    df_horse_grouped.rename(columns={col: col + '_mean'}, inplace=True)

df_data = df_data.merge(df_jockey_grouped, on=['Date', 'RaceNumber'], how='left')
df_data = df_data.merge(df_horse_grouped, on=['Date', 'RaceNumber'], how='left')


# delta between opponent mean embedding and the horse/jockey embedding
for col in j_emb_cols:
    df_data[col + '_delta_mean'] = df_data[col] - df_data[col + '_mean']

for col in h_emb_cols:
    df_data[col + '_delta_mean'] = df_data[col] - df_data[col + '_mean']

# Append Max opponent embeddings
df_jockey_grouped = df_jockey.groupby(['Date', 'RaceNumber']).max().reset_index()
for col in j_emb_cols:
    df_jockey_grouped.rename(columns={col: col + '_max'}, inplace=True)

df_horse_grouped = df_horse.groupby(['Date', 'RaceNumber']).max().reset_index()
for col in h_emb_cols:
    df_horse_grouped.rename(columns={col: col + '_max'}, inplace=True)

df_data = df_data.merge(df_jockey_grouped, on=['Date', 'RaceNumber'], how='left')
df_data = df_data.merge(df_horse_grouped, on=['Date', 'RaceNumber'], how='left')

# delta between opponent mean embedding and the horse/jockey embedding
for col in j_emb_cols:
    df_data[col + '_delta_max'] = df_data[col] - df_data[col + '_max']

for col in h_emb_cols:
    df_data[col + '_delta_max'] = df_data[col] - df_data[col + '_max']


# Append Max opponent embeddings
df_jockey_grouped = df_jockey.groupby(['Date', 'RaceNumber']).min().reset_index()
for col in j_emb_cols:
    df_jockey_grouped.rename(columns={col: col + '_min'}, inplace=True)

df_horse_grouped = df_horse.groupby(['Date', 'RaceNumber']).min().reset_index()
for col in h_emb_cols:
    df_horse_grouped.rename(columns={col: col + '_min'}, inplace=True)

df_data = df_data.merge(df_jockey_grouped, on=['Date', 'RaceNumber'], how='left')
df_data = df_data.merge(df_horse_grouped, on=['Date', 'RaceNumber'], how='left')

# delta between opponent mean embedding and the horse/jockey embedding
for col in j_emb_cols:
    df_data[col + '_delta_min'] = df_data[col] - df_data[col + '_min']

for col in h_emb_cols:
    df_data[col + '_delta_min'] = df_data[col] - df_data[col + '_min']




print(f"Number of rows in df_data: {len(df_data)}")
df_data= df_data[df_data['J_Emb_0'].notna()]


print(f"Number of rows in df_data after: {len(df_data)}")

df_data['top_three'] = df_data['Placing'].apply(lambda x: 1 if x <= 3 else 0)

keep_cols = [
            'Date', 
            'RaceNumber', 
            'Horse',
            'Placing',
            'top_three',
            'Placing_TR1',	
            'Placing_TR2',	
            'Placing_TR3',	
            'Win Odds',
            'Dr.', 
            # 'J_Emb_48_delta_max',
            # 'J_Emb_47_delta_max',
            # 'J_Emb_0_delta_max',
            # 'J_Emb_1_delta_max',
            # 'J_Emb_3_delta_max',

            # 'J_Emb_21_delta_mean',
            # 'J_Emb_24_delta_mean',

            # 'H_Emb_21_delta_mean',
            # 'H_Emb_25_delta_mean',

            # 'H_Emb_46_delta_max',
            # 'H_Emb_0_delta_max',

            # 'H_Emb_49_delta_min', 

             ]

numeric_cols = [

            # 'Act. Wt.',
            # 'Declar. Horse Wt.',
            # 'DistanceMeter',
            'Placing_TR1',	
            'Placing_TR2',	
            'Placing_TR3',	
            'Placing_TR4',	
            'Placing_TR5',	
            'Placing_TR6',	
            'Placing_TR7',	
            'Placing_TR8',	
            'Placing_TR9',	
            'Placing_TR10',
]

cols = df_data.columns.tolist()

embedding_cols = []
embedding_cols = h_emb_cols + j_emb_cols
# embedding_cols = [col for col in cols if col[0:6] in ['H_Emb_']]

emb_delta_max_opp_cols = [col for col in cols if '_delta_max' in col]
emb_delta_min_opp_cols = [col for col in cols if '_delta_min' in col]
emb_delta_mean_opp_cols = [col for col in cols if '_delta_mean' in col]
emb_mean_opp_cols = [col for col in cols if '_mean' in col]
emb_max_opp_cols = [col for col in cols if '_max' in col]
emb_min_opp_cols = [col for col in cols if '_min' in col]



# print(f"Number of embedding columns: {len(embedding_cols)}")
# print(f"Number of numeric columns: {len(numeric_cols)}")
# print(f"Number of emb_delta_opp columns: {len(emb_delta_opp_cols)}")
# print(f"Number of emb_mean_opp columns: {len(emb_mean_opp_cols)}")
# print(f"Number of emb_max_opp columns: {len(emb_max_opp_cols)}")
# print(f"Number of emb_min_opp columns: {len(emb_min_opp_cols)}")



best_iteration = [0, 0, 0, 0, 0]
best_features = []
for i in range(0, 1):

    embedding_cols_rand = []
    embedding_cols_rand = random.sample(embedding_cols, 10)
    numeric_cols_rand = []
    numeric_cols_rand = random.sample(numeric_cols, 10)
    
    emb_delta_max_opp_cols_rand = random.sample(emb_delta_max_opp_cols, 50)
    emb_delta_min_opp_cols_rand = random.sample(emb_delta_min_opp_cols, 50)
    emb_delta_mean_opp_cols_rand = random.sample(emb_delta_mean_opp_cols, 10)
    emb_mean_opp_cols_rand = random.sample(emb_mean_opp_cols, 10)
    emb_max_opp_cols_rand = random.sample(emb_max_opp_cols, 50)
    emb_min_opp_cols_rand = random.sample(emb_min_opp_cols, 50)

    #  embedding_cols_rand + 
    features = set(keep_cols
                #    + numeric_cols_rand
                   + embedding_cols_rand
                #    + emb_delta_max_opp_cols_rand 
                #    + emb_delta_min_opp_cols_rand 
                #    + emb_delta_mean_opp_cols_rand 
                   + emb_min_opp_cols_rand
                #    + emb_mean_opp_cols_rand 
                   + emb_max_opp_cols_rand
                   )
    features = list(features)

    df_encoded = df_data[features].copy()
    df_train = df_encoded[df_encoded['Date'] < '2025-03-22'].copy()
    df_test = df_encoded[df_encoded['Date'] >= '2025-03-22'].copy()

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

    search_params = {
        # 'objective': 'multi:softprob',  # Multi-class classification
        # 'num_class': 5,  # Adjust based on the number of classes in your target variable
        # 'device': 'cuda:0',  # Use GPU
        # 'tree_method': 'gpu_hist',  # Use GPU for training
        # 'predictor': 'gpu_predictor',  # Use GPU for prediction
        'eval_metric': ['mlogloss'],

        'max_depth': [5, 7],
        'min_child_weight': [2],
        'gamma': [0.45, 0.5, 0.55],
        'subsample': [0.8],
        'colsample_bytree': [.95, .98, 1.0],
        'learning_rate':[0.12, 0.125, 0.13],
        'n_estimators': [100, 110, 120]
    }


    #     'eval_metric': 'mlogloss',
    params = {

        'objective': 'binary:logistic',
        'max_depth': 4,
        'min_child_weight': 1,
        'gamma': .55,
        'subsample': 0.85,
        'colsample_bytree': 1,
        'learning_rate':0.01, 
        'n_estimators':1000,
        # 'early_stopping_rounds': 100,

    }


    model = XGBClassifier(**params)
    model.fit(X_train_drop, y_train)

    # model =  XGBClassifier()
    # grid_search = GridSearchCV(model, search_params, cv=3, scoring='accuracy', verbose=1)
    # grid_search.fit(X_train_drop , y_train)
    # print(grid_search.best_params_)
    # model = XGBClassifier(**grid_search.best_params_)
    # model.fit(X_train_drop, y_train)

    print('Finished Training XGBoost model with GPU support...')
    #preds = model.predict(dtest)
    y_probs = model.predict_proba(X_test_drop)[:, 1]
    preds = (y_probs > 0.5).astype(int)

    # df = pd.DataFrame(dtest.get_data(), columns=X_test.columns)
    df_preds = pd.DataFrame(preds, columns=['Predicted Placing'])
    df_preds['top_three'] = y_test.values

    df_merged = pd.concat([X_test[['Horse', 'Date', 'RaceNumber', 'Placing']], df_preds], axis=1)

    mean_absolute_error = 0.0

    # print(f'mean_absolute_error: {mean_absolute_error:.2f}')

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)

    # print(f'Accuracy: {accuracy:.2f}')
    # print(f'Precision: {precision:.2f}')
    # print(f'Recall: {recall:.2f}')

    print(f'iteration: {i}')
    if accuracy > best_iteration[0]:
        best_iteration = [accuracy, precision, recall, accuracy, mean_absolute_error]
        best_features = features
        print(f'New best iteration: {i}')
        print(f'Precision: {precision:.2f}, Recall: {recall:.2f}, Accuracy: {accuracy:.2f}, MAE: {mean_absolute_error:.2f}')
        print(f'Features: {features}')


    plot_importance(model, importance_type='gain')
    plt.show()





# trailing_avg_cols = []
# # list_tr = ['TR1','TR2','TR3','TR4','TR5']
# # trailing_avg_cols = [col for col in cols if col[-3:] in list_tr\
# #                         and 'Placing' in col\
# #                         and ('Score range' in col\
# #                         or 'DistanceMeterAsStr' in col)]


    # features = ['Date', 'RaceNumber', 'Horse', 'Placing', 'top_three','Horse', 'Dr.', 
    #                      'J_Emb_47', 'J_Emb_4', 'H_Emb_29', 'Placing_TR8', 
    #                      'J_Emb_21',  'H_Emb_3', 'J_Emb_48', 'Placing_TR10', 'H_Emb_28', 
    #                      'J_Emb_13', 'H_Emb_10', 'J_Emb_28', 'Placing_TR2', 'H_Emb_22', 
    #                      'Placing_TR1', 'H_Emb_7', 'Placing_TR9', 'J_Emb_8'                             
    #                      ]