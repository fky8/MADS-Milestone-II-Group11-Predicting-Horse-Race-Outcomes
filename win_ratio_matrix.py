
# .\.venv\Scripts\Activate.ps1
import numpy as np
from collections import defaultdict
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def get_race_dataset(start_date='2019-01-01', end_date='2025-12-31'):   
    """
    gets horse race dataset and applies set filters to standardize the race data
    accross functions.  
    """

    df = pd.read_csv('./data/Race_comments_gear_horse_competitors_2019_2025.csv')
    #  filter out horses without a placing due to withdrawal or disqualification
    df['Placing'] = pd.to_numeric(df['Placing'], errors='coerce')
    df_filtered = df[df['Placing'].notna()]    
    df_filtered['Date'] = pd.to_datetime(df_filtered['Date'])

    return df_filtered[(df_filtered['Date'] >= start_date) & (df_filtered['Date'] <= end_date)]



def create_win_ratio_matrix(df_race_dataset=None):
    """
    Create a win ratio matrix for horses based on horse vs horse.
    """

    horses = df_race_dataset.sort_values(by=['Horse'], ascending=[True])['Horse'].unique().tolist()
    df_horse_race_outcomes = df_race_dataset.sort_values(by=['Date', 'RaceNumber'], ascending=[True, True])[['Horse', 'Date', 'RaceNumber', 'Placing']]
    list_horse_race_outcomes = df_horse_race_outcomes.values.tolist()

    list_races = df_horse_race_outcomes[['Date', 'RaceNumber']].drop_duplicates().values.tolist()
    dict_races = {(race[0], race[1]): [] for race in list_races}

    for row in list_horse_race_outcomes:
        dict_races[(row[1], row[2])].append(row) 

    dict_horse_occurrence = defaultdict(float)
    dict_horse_won = defaultdict(float)
    dict_horse_win_ratio = defaultdict(float)
    for horse in horses:
        for opponent in horses:
            if horse != opponent:  
                dict_horse_occurrence[(horse, opponent)] = 0.0
                dict_horse_won[(horse, opponent)] = 0.0
                dict_horse_win_ratio[(horse, opponent)] = 0.0

    for race in list_races:        
        for outcome in dict_races[(race[0], race[1])]:            
            for outcome_opponent in dict_races[(race[0], race[1])]:                
                if outcome[0] != outcome_opponent[0]:                    
                    points = outcome_opponent[3] - outcome[3]
                    dict_horse_won[(outcome[0], outcome_opponent[0])] += points
                    dict_horse_occurrence[(outcome[0], outcome_opponent[0])] += 1.0
                    
                    points = dict_horse_won[(outcome[0], outcome_opponent[0])]
                    occurences = dict_horse_occurrence[(outcome[0], outcome_opponent[0])]
                    if occurences > 0:
                        dict_horse_win_ratio[(outcome[0], outcome_opponent[0])] = points / occurences                 

    df_win_ratio = pd.DataFrame.from_dict(dict_horse_win_ratio, orient='index', columns=['Win Ratio'])
    df_win_ratio.reset_index(inplace=True)
    df_win_ratio[['Horse', 'Opponent']] = pd.DataFrame(df_win_ratio['index'].tolist(), index=df_win_ratio.index)
    df_win_ratio.drop('index', axis=1, inplace=True)
    df_win_ratio.sort_values(by=['Horse', 'Opponent'], ascending=[True, True], inplace=True)

    # for index, row in df_win_ratio.head(1000).iterrows():
    #     print(f"{row['Horse']} vs {row['Opponent']}: {row['Win Ratio']:.2f}")

    # Normalize the win ratio to a range of 0 to 1 to remove negative values
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df_win_ratio[['Win Ratio']])
    df_win_ratio_normalized = pd.DataFrame(scaled_values, columns=['Win Ratio Normalized'])
    df_win_ratio = pd.concat([df_win_ratio, df_win_ratio_normalized], axis=1)

    # for index, row in df_win_ratio.head(1000).iterrows():
    #     print(f"{row['Horse']} vs {row['Opponent']}: {row['Win Ratio Normalized']:.2f}")

    # df_win_ratio = df_win_ratio[df_win_ratio['Win Ratio Normalized'] > 0.0]
    # handle cases where normalized win ratio is zero
    # print(f"Total horse pairs: {df_win_ratio[df_win_ratio['Win Ratio Normalized'] == 0.0].head(12)}")
    
    df_win_ratio['Win Ratio Normalized'] = df_win_ratio['Win Ratio Normalized'].replace(0.0, 0.0001)
    # df_win_ratio['Win Ratio Normalized'] = df_win_ratio['Win Ratio Normalized'].replace(0.5, 0.0001)
    
    for index, row in df_win_ratio.iterrows():
        key = (row['Horse'], row['Opponent'])
        # dict_horse_win_ratio[key] = row['Win Ratio']
        dict_horse_win_ratio[key] = row['Win Ratio Normalized']
        # print(f"{row['Horse']} vs {row['Opponent']}: {dict_horse_win_ratio[key]:.2f}")


    return dict_horse_win_ratio


def create_race_embeddings(df_race_dataset=None, dict_win_ratio=None):

    horses = df_race_dataset.sort_values(by=['Horse'], ascending=[True])['Horse'].unique().tolist()

    embedding_dim = 10
    race_embeddings = {
        horse: np.random.randn(embedding_dim) for horse in horses
    }

    # for embedding in race_embeddings.values():
    #     print(f"Initial embedding: {embedding}")

    learning_rate = 0.01
    num_epochs = 100

    for epoch in range(num_epochs):
        total_loss = 0
        for (i, j), ratio in dict_win_ratio.items():
            dot_product = np.dot(race_embeddings[i], race_embeddings[j])
            diff = dot_product - np.log(ratio)
            total_loss += 0.5 * diff**2
            gradient = diff * race_embeddings[j]
            race_embeddings[i] -= learning_rate * gradient
            # print(f"Comparing {i} vs {j}: dot_product={dot_product}")
            # print(f"Comparing {i} vs {j}: ratio={ratio}")
            # print(f"Comparing {i} vs {j}: diff={diff}")
            # print(f"Comparing {i} vs {j}: total_loss={total_loss}")
            # print(f"Comparing {i} vs {j}: gradient={gradient}")
            # print(f"Comparing {i} vs {j}: race_embeddings={race_embeddings[i]}")



        print(f"Epoch: {epoch+1}, Loss: {total_loss}, {total_loss / len(dict_win_ratio)}")

    # print("Final horse embeddings:")
    # for horse, embedding in race_embeddings.items(): 
    #     print(f"{horse}: {embedding}")

    return race_embeddings


df_rd = get_race_dataset(start_date='2024-05-01', end_date='2024-5-31')
# df_rd = df_rd[(df_rd['RaceNumber'] == 1.0) & (df_rd['Date'] == '2024-05-01')]
# print(len(df_rd))
# print(df_rd[['Date', 'RaceNumber']].head(len(df_rd)))
dict_wr = create_win_ratio_matrix(df_race_dataset=df_rd)

# for key, value in dict_wr.items():
#     print(f"{key[0]} vs {key[1]}: {value:.2f}")

race_embeddings = create_race_embeddings(df_race_dataset=df_rd, dict_win_ratio=dict_wr)



    # dict_possible_points = {}
    # total = 0
    # for i in range(1, 31):
    #     total += i 
    #     dict_possible_points[i] = total - i