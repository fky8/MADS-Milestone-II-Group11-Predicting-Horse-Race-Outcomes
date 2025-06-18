import xgboost as xgb
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.dirname(current_dir)
sys.path.append(scripts_dir)
import query_repository as qr
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error

trs = ['']
df_data = qr.get_ml_training_data(trs)


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
df_encoded = df_data.drop(columns=drop_cols, errors='ignore')


df_races = df_encoded[['Date', 'RaceNumber']].drop_duplicates()
df_sample_races = df_races.sample(frac=0.20, random_state=777, replace=False)
df_merged = pd.merge(df_encoded, df_sample_races, on =['Date', 'RaceNumber'], how='left', indicator=True)
df_train = df_merged[df_merged['_merge'] == 'left_only'].drop('_merge', axis=1)
df_test = pd.merge(df_encoded, df_sample_races, on =['Date', 'RaceNumber'], how='right', indicator=True)

# print(len(df_train), len(df_test), len(df_encoded), len(df_train) + len(df_test))
X_train = df_train.drop(columns=['Placing', 'Finish Time In Seconds'])
X_test = df_test.drop(columns=['Placing', 'Finish Time In Seconds', '_merge'])
y_train = df_train['Placing']
y_test = df_test['Placing']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=777)


# 'Date',
# 'RaceNumber',
# 'Horse',

X_train_drop = X_train.drop(columns=['Date', 'RaceNumber', 'Horse'], errors='ignore')
X_test_drop = X_test.drop(columns=['Date', 'RaceNumber', 'Horse'], errors='ignore')


# print(X_test_drop.dtypes)


print('1')
dtrain = xgb.DMatrix(X_train_drop, label=y_train)
dtest = xgb.DMatrix(X_test_drop, label=y_test)
print('2')
params = {
    # 'objective': 'multi:softprob',  # Multi-class classification
    # 'num_class': 5,  # Adjust based on the number of classes in your target variable
    # 'device': 'cuda:0',  # Use GPU
    # 'tree_method': 'gpu_hist',  # Use GPU for training
    # 'predictor': 'gpu_predictor',  # Use GPU for prediction
    'eval_metric': 'mlogloss',
    'n_estimators': 100,
    'learning_rate': 0.1,
}
print('Training XGBoost model with GPU support...')
model = xgb.train(params, dtrain, num_boost_round=100)
print('Finished Training XGBoost model with GPU support...')
preds = model.predict(dtest)

# df = pd.DataFrame(dtest.get_data(), columns=X_test.columns)
df_preds = pd.DataFrame(preds, columns=['Predicted Placing'])
df_preds['Actual Placing'] = y_test.values

df_merged = pd.concat([X_test[['Horse', 'Date', 'RaceNumber']], df_preds], axis=1)

df_merged['Predicted Placing Rank'] = df_merged.groupby(['Date', 'RaceNumber'])['Predicted Placing'].rank(method='average', ascending=True)

print(df_merged[['Horse', 'Date', 'RaceNumber', 'Actual Placing', 'Predicted Placing', 'Predicted Placing Rank']].sort_values(by=['Date','RaceNumber','Predicted Placing Rank'], ascending=[True,True,True]).head(24))
print('finished')

print('y_test', type(y_test))
print('preds', type(preds))

# df_merged = df_merged[df_merged['Actual Placing'] >= 3].sort_values(by=['Date', 'RaceNumber'], ascending=[True, True])

# df_merged['Predicted Placing Rank'] = df_merged['Predicted Placing Rank'].apply(lambda x: 1 if not pd.isna(x) and x >= 3 else 0)
# df_merged['Actual Placing'] = df_merged['Actual Placing'].apply(lambda x: 1 if not pd.isna(x) and x >= 3 else 0)

mean_absolute_error = mean_absolute_error(df_merged['Actual Placing'], 
                          df_merged['Predicted Placing Rank'])


for col in X_train_drop.columns:
    print(f'{col}: {X_train_drop[col].dtype}')

print(f'mean_absolute_error: {mean_absolute_error:.2f}')



