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

# df_data = pd.read_csv('./data/ml_dataset_2019_2025_jockey_2147.csv')
df_data = pd.read_csv('./data/ml_dataset_2019_2025_jockey_placing_2147.csv')

print(f"Number of rows in df_data: {len(df_data)}")
df_data= df_data[df_data['J_Emb_0'].notna()]
print(f"Number of rows in df_data after: {len(df_data)}")

df_data['top_three'] = df_data['Placing'].apply(lambda x: 1 if x <= 3 else 0)

# 'Date',
# 'RaceNumber',
# 'Horse',

drop_cols = [
             'Jockey', 
             'Score range',
             'Going',
             'Win Odds',
             'Course Detail',
             'Country',
             'Colour',
             'Sex',
             'Import Type',
             'Handicap'
             ]


keep_cols = [
            'Date', 
            'RaceNumber', 
            'Horse',
            'Placing',
            'top_three',

           'Dr.', 
             'Act. Wt.',
             'Declar. Horse Wt.',
             'DistanceMeter'

             ]






embedding_cols = []
cols = df_data.columns.tolist()
embedding_cols = [col for col in cols if col[0:6] in ['H_Emb_', 'J_Emb_']]


list_tr = ['TR1','TR2','TR3','TR4','TR5']
trailing_avg_cols = [col for col in cols if col[-3:] in list_tr\
                        and 'Placing' in col\
                        and ('Score range' in col\
                        or 'DistanceMeterAsStr' in col)]


for col in trailing_avg_cols:
  df_data[col] = np.log1p(df_data[col].astype(float))

# df_encoded = df_data.drop(columns=drop_cols, errors='ignore')

df_encoded = df_data[keep_cols + trailing_avg_cols + embedding_cols]
print("columns", df_encoded.columns.tolist())

df_races = df_encoded[['Date', 'RaceNumber']].drop_duplicates()
df_sample_races = df_races.sample(frac=0.20, random_state=777, replace=False)
df_merged = pd.merge(df_encoded, df_sample_races, on =['Date', 'RaceNumber'], how='left', indicator=True)
df_train = df_merged[df_merged['_merge'] == 'left_only'].drop('_merge', axis=1)
df_test = pd.merge(df_encoded, df_sample_races, on =['Date', 'RaceNumber'], how='right', indicator=True)

# print(len(df_train), len(df_test), len(df_encoded), len(df_train) + len(df_test))
X_train = df_train.drop(columns=['top_three'])
X_test = df_test.drop(columns=['top_three',  '_merge'])
y_train = df_train['top_three']
y_test = df_test['top_three']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=777)


# 'Date',
# 'RaceNumber',
# 'Horse',

X_train_drop = X_train.drop(columns=['Placing', 'Date', 'RaceNumber', 'Horse'], errors='ignore')
X_test_drop = X_test.drop(columns=['Placing', 'Date', 'RaceNumber', 'Horse'], errors='ignore')


# print(X_test_drop.dtypes)


print('1')
dtrain = xgb.DMatrix(X_train_drop, label=y_train)
dtest = xgb.DMatrix(X_test_drop, label=y_test)
print('2')
search_params = {
    # 'objective': 'multi:softprob',  # Multi-class classification
    # 'num_class': 5,  # Adjust based on the number of classes in your target variable
    # 'device': 'cuda:0',  # Use GPU
    # 'tree_method': 'gpu_hist',  # Use GPU for training
    # 'predictor': 'gpu_predictor',  # Use GPU for prediction
    'eval_metric': ['mlogloss'],

    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],

}


#     'eval_metric': 'mlogloss',
params = {

    'objective': 'binary:logistic',
    'max_depth': 3,
    'min_child_weight': 1,
    'gamma': .5,
    'subsample': 0.8,
    'colsample_bytree': 1,
    'learning_rate':0.1, 
    'n_estimators':100

}


model = XGBClassifier(**params)
model.fit(X_train_drop, y_train)

# model =  XGBClassifier(learning_rate=0.1, n_estimators=100)
# grid_search = GridSearchCV(model, search_params, cv=3, scoring='accuracy', verbose=1)
# grid_search.fit(X_train_drop , y_train)
# print(grid_search.best_params_)
# model = xgb.train(params, dtrain, num_boost_round=100)

print('Finished Training XGBoost model with GPU support...')
#preds = model.predict(dtest)
y_probs = model.predict_proba(X_test_drop)[:, 1]
preds = (y_probs > 0.5).astype(int)

# df = pd.DataFrame(dtest.get_data(), columns=X_test.columns)
df_preds = pd.DataFrame(preds, columns=['Predicted Placing'])
df_preds['top_three'] = y_test.values

df_merged = pd.concat([X_test[['Horse', 'Date', 'RaceNumber', 'Placing']], df_preds], axis=1)

# df_merged['Predicted Placing Rank'] = df_merged.groupby(['Date', 'RaceNumber'])['Predicted Placing'].rank(ascending=False, method='first')

print(df_merged[['Horse', 'Date', 'RaceNumber', 'Placing', 'top_three', 'Predicted Placing']].sort_values(by=['Date','RaceNumber','Placing'], ascending=[True,True,True]).head(24))
print('finished')

print('y_test', type(y_test))
print('preds', type(preds))

# df_merged = df_merged[df_merged['Actual Placing'] <= 3].sort_values(by=['Date', 'RaceNumber'], ascending=[True, True])

print("After filtering for actual placing >= 3:")
print(df_merged[['Horse', 'Date', 'RaceNumber', 'Placing', 'top_three', 'Predicted Placing']].sort_values(by=['Date','RaceNumber','Placing'], ascending=[True,True,True]).head(24))

mean_absolute_error = mean_absolute_error(df_merged['top_three'], 
                          df_merged['Predicted Placing'])

print(f'mean_absolute_error: {mean_absolute_error:.2f}')

accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')


plot_importance(model, importance_type='gain')
plt.show()