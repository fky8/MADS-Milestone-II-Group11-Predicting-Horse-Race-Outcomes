import pandas as pd
import numpy as np
df = pd.read_csv('data\jockey\jockey_race_data_clean.csv')
df = df.dropna(subset=['Placing'])
df['Top3'] = df['Placing'].str.extract(r'^(\d+)/')[0].astype(int).le(3).astype(int)
df_clean = df.drop(columns=['Placing', 'Unnamed: 13', 'race_wins', 'first_place_win', 'second_place_win', 'third_place_win', 'Race Index', 'Race Sub Index', 'race_description', 'race_course', 'race_date'])

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from sklearn.metrics import classification_report


unique_races = df['Race Index'].unique()
train_races, test_races = train_test_split(unique_races, test_size=0.2, random_state=42)

train_df = df_clean[df['Race Index'].isin(train_races)]
test_df = df_clean[df['Race Index'].isin(test_races)]


X_train = train_df.drop(columns=['Top3'])
y_train = train_df['Top3']

X_test = test_df.drop(columns=['Top3'])
y_test = test_df['Top3']


categorical_cols = X_train.select_dtypes(include='object').columns.tolist()
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', SimpleImputer(strategy='mean'), numeric_cols)
])


pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])


pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

from collections import defaultdict

ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat']
ohe_feature_names = ohe.get_feature_names_out(categorical_cols)

all_feature_names = np.concatenate([ohe_feature_names, numeric_cols])

coefs = pipeline.named_steps['classifier'].coef_[0]

importance_by_column = defaultdict(float)

for feature, coef in zip(all_feature_names, coefs):

    orig_col = feature.split('_')[0]
    importance_by_column[orig_col] += abs(coef)

importance_df = pd.DataFrame({
    'Column': list(importance_by_column.keys()),
    'Total_Importance': list(importance_by_column.values())
}).sort_values(by='Total_Importance', ascending=False)

print(importance_df)

from sklearn.ensemble import RandomForestClassifier


unique_races = df['Race Index'].unique()
train_races, test_races = train_test_split(unique_races, test_size=0.2, random_state=42)

train_df = df_clean[df['Race Index'].isin(train_races)]
test_df = df_clean[df['Race Index'].isin(test_races)]

X_train = train_df.drop(columns=['Top3'])
y_train = train_df['Top3']

X_test = test_df.drop(columns=['Top3'])
y_test = test_df['Top3']

categorical_cols = X_train.select_dtypes(include='object').columns.tolist()
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', SimpleImputer(strategy='mean'), numeric_cols)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat']
ohe_feature_names = ohe.get_feature_names_out(categorical_cols)
all_feature_names = np.concatenate([ohe_feature_names, numeric_cols])

importances = pipeline.named_steps['classifier'].feature_importances_

importance_by_column = defaultdict(float)

for feature, imp in zip(all_feature_names, importances):
    base_col = feature.split('_')[0]
    importance_by_column[base_col] += imp

importance_df = pd.DataFrame({
    'Column': list(importance_by_column.keys()),
    'Total_Importance': list(importance_by_column.values())
}).sort_values(by='Total_Importance', ascending=False)

print(importance_df)