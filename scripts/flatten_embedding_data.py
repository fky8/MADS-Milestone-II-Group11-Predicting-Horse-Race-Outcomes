"""
Takes an embedding CSV file with embeddings created as a weekly interval.
Embeddings were calculated in a weekly interval to prevent data leakage, as 
not to use embedding relationships from future races to predict races in the past.
Additionally, embeddings were calculated on a weekly basis to save time and memory.

However, to merge the embeddings with other race data, we needed to flatten the embeddings
because race data needed join "Date" on "Date" is between "Start Date" and "End Date" of the embedding.
To achieve this, a cross join is needed, but given the size of the data (300,000+ rows), our computers
ran out of memory. To improve this, we flatten the embeddings to have a row for each day

Ex. Embedding 1 represents 100 rows of data between 5/1/2025 and 5/7/2025. 
After flattening, the emebedding have 100 rows x 7 days.


"""

import os
import pandas as pd
from datetime import datetime, timedelta

df = pd.read_csv('./data/Horse_Horse_Embeddings_180.csv')

list_dates = df['Date End'].unique().tolist()[::-1]

# print(list_dates)

list_dfs = []
for date in list_dates:  # Limiting to first 10 dates for demonstration
    for i in range(0, 7):
        # next_date = datetime.strptime(date, '%Y-%m-%d') - timedelta(days=i)
        new_date = pd.to_datetime(date) + timedelta(days=i)
        print(f"Processing date: {new_date.strftime('%Y-%m-%d')}")
        print(f"Processing end date: {pd.to_datetime(date).strftime('%Y-%m-%d')}")
        df_next = df[df['Date End'] == pd.to_datetime(date).strftime('%Y-%m-%d')]
        df_next.drop(columns=['Date End', 'Date Begin'], inplace=True, errors='ignore')
        df_next['Date'] = new_date
        
        list_dfs.append(df_next)


df_flattened = pd.concat(list_dfs, ignore_index=True)
df_flattened = df_flattened.sort_values(by=['Date'], ascending=True)

for i in range(0, 50):
    df_flattened.rename(columns={f'{str(i)}': f'H_Emb_{str(i)}'}, inplace=True, errors='ignore')

script_dir = os.path.dirname(os.path.abspath(__file__)) 
file_path = os.path.join(script_dir, f"../data/Horse_Horse_Embeddings_180_flat.csv")
df_flattened.to_csv(file_path, index=False)