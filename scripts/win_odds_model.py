import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score



df = pd.read_csv('./data/Race_comments_gear_horse_competitors_2019_2025.csv')
df['Placing'] = pd.to_numeric(df['Placing'], errors='coerce')
df_filtered = df[df['Placing'].notna()]
df_filtered['Date'] = pd.to_datetime(df_filtered['Date'])

df_filtered = df_filtered[df_filtered['Date'] >= '2019-01-01']

df_filtered = df_filtered[df_filtered['Date'] >= '2025-03-22']
df_filtered = df_filtered[df_filtered['Date'] < '2025-05-22']


df_filtered['Win Odds'] = df_filtered['Win Odds'].astype(float)
df_filtered = df_filtered[['Date', 'RaceNumber', 'Horse', 'Placing', 'Win Odds']]

df_filtered['top_three'] = df_filtered['Placing'].apply(lambda x: 1 if x <= 3 else 0)

df_filtered = df_filtered.sort_values(by=['Date', 'RaceNumber', 'Win Odds'], 
                                      ascending=[True, True, True])

df_filtered['Rank'] = df_filtered.groupby(['Date', 'RaceNumber'])\
    ['Win Odds'].rank(method='first', ascending=True)

df_filtered = df_filtered[df_filtered['Rank'].notna()]

df_filtered['Rank'] = df_filtered['Rank'].astype(int)
df_filtered['pred_top_three'] = df_filtered['Rank'].apply(lambda x: 1 if x <= 3 else 0)


print(df_filtered.head(10))

y = df_filtered['top_three']
preds = df_filtered['pred_top_three']

accuracy = accuracy_score(y, preds)
precision = precision_score(y, preds)
recall = recall_score(y, preds)

print(f'Precision: {precision:.2f}, Recall: {recall:.2f}, Accuracy: {accuracy:.2f}')