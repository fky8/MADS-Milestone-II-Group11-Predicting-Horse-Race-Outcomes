import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt


jockey_race_data = pd.read_csv('data/jockey/jockey_race_data_clean.csv')
weather_data = pd.read_csv('data/weather_data/race_date_with_weather.csv')


weather_data['Date'] = pd.to_datetime(weather_data['Date'], errors='coerce')
jockey_race_data['race_date'] = pd.to_datetime(jockey_race_data['race_date'], errors='coerce')


jockey_race_data = jockey_race_data.dropna(subset=['Placing'])
jockey_race_data['Top3'] = (
    jockey_race_data['Placing']
    .str.extract(r'^(\d+)/')[0]
    .astype(int)
    .le(3)
    .astype(int)
)

jockey_race_data['NumericPlacing'] = (
    jockey_race_data['Placing']
    .str.extract(r'^(\d+)/')[0]
    .astype(float)
)

jockey_race_data = jockey_race_data.sort_values(by=['Horse', 'race_date'])

jockey_race_data['last_placement'] = (
    jockey_race_data.groupby('Horse')['NumericPlacing']
    .shift(1)
    .fillna('first')
)

jockey_race_data['days_since_last_race'] = (
    jockey_race_data.groupby('Horse')['race_date']
    .diff()
    .dt.days
    .fillna('first')
)

jockey_race_data = jockey_race_data.drop(columns=['NumericPlacing', 'Placing', 'Unnamed: 13', 'race_wins', 'first_place_win','second_place_win', 'third_place_win', 'Race Index', 'Race Sub Index','Rtg.'], errors='ignore')
weather_data = weather_data.drop(columns=['MaximumRelativeHumidity', 'MinimumRelativeHumidity', 'GrassMinimumTemperature', 'Rainfall(mm)', 'MeanUVIndex', 'MaxUVIndex'])


X = jockey_race_data.copy()


X = pd.merge(X, weather_data, how='left', left_on='race_date', right_on='Date')


X['race_date'] = pd.to_datetime(X['race_date'])
X = X.drop(columns=['Date_x', 'Date_y'], errors='ignore')


y = X['Top3']
X = X.drop(columns=['Top3'])


scores = []
all_preds = []
all_truths = []


feature_importance_accum = defaultdict(float)
feature_names = None


imputer = SimpleImputer(strategy='constant', fill_value=-1)


unique_dates = sorted(X['race_date'].dropna().unique())

for test_date in tqdm(unique_dates):
    train_mask = X['race_date'] < test_date
    test_mask = X['race_date'] == test_date

    if train_mask.sum() < 100 or test_mask.sum() == 0:
        continue

    X_train = X[train_mask].drop(columns=['race_date'], errors='ignore')
    y_train = y[train_mask]
    X_test = X[test_mask].drop(columns=['race_date'], errors='ignore')
    y_test = y[test_mask]


    dt_cols = X_train.select_dtypes(include='datetime64').columns
    X_train = X_train.drop(columns=dt_cols, errors='ignore')
    X_test = X_test.drop(columns=dt_cols, errors='ignore')

    X_train_enc = pd.get_dummies(X_train)
    X_test_enc = pd.get_dummies(X_test)
    X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

    X_train_enc = pd.DataFrame(imputer.fit_transform(X_train_enc), columns=X_train_enc.columns, index=X_train_enc.index)
    X_test_enc = pd.DataFrame(imputer.transform(X_test_enc), columns=X_test_enc.columns, index=X_test_enc.index)


    model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train_enc, y_train)


    if feature_names is None:
        feature_names = X_train_enc.columns.tolist()


    for name, score in zip(feature_names, model.feature_importances_):
        feature_importance_accum[name] += score


    y_pred = model.predict(X_test_enc)
    all_preds.extend(y_pred)
    all_truths.extend(y_test)
    score = f1_score(y_test, y_pred, zero_division=0)
    scores.append({'date': test_date, 'f1': score})


results_df = pd.DataFrame(scores)
print(results_df.describe())
print(results_df)


importance_df = pd.DataFrame({
    'feature': list(feature_importance_accum.keys()),
    'total_importance': list(feature_importance_accum.values())
})
importance_df['avg_importance'] = importance_df['total_importance'] / len(results_df)
importance_df = importance_df.sort_values(by='avg_importance', ascending=False)

print(importance_df.head(20))

top_n = 20
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'].head(top_n)[::-1], importance_df['avg_importance'].head(top_n)[::-1])
plt.xlabel("Average Importance")
plt.title(f"Top {top_n} Features Across All Days")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(results_df['date'], results_df['f1'], marker='o', linestyle='-')
plt.title('Daily F1 Score Over Time')
plt.xlabel('Race Date')
plt.ylabel('F1 Score')
plt.grid(True)
plt.tight_layout()
plt.show()

