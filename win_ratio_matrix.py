
# .\.venv\Scripts\Activate.ps1
import numpy as np
from collections import defaultdict
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def create_win_ratio_matrix():
    """
    Create a win ratio matrix for horses based on horse vs horse.
    """

    df = pd.read_csv('./data/Race_comments_gear_horse_competitors_2019_2025.csv')
    print("df row count start: ", len(df))
    #  filter out horses without a placing due to withdrawal or disqualification
    df['Placing'] = pd.to_numeric(df['Placing'], errors='coerce')
    df_filtered = df[df['Placing'].notna()]    
    print("df row count filter non placing horses: ", len(df_filtered))
    df_filtered['Date'] = pd.to_datetime(df_filtered['Date'])
    print("df row count filter non placing horses: ", len(df_filtered))


    horses = df_filtered.sort_values(by=['Horse'], ascending=[True])['Horse'].unique().tolist()

    df_horse_race_outcomes = df_filtered.sort_values(by=['Date', 'RaceNumber'], ascending=[True, True])[['Horse', 'Date', 'RaceNumber', 'Placing']]
    list_horse_race_outcomes = df_horse_race_outcomes.values.tolist()

    list_races = df_horse_race_outcomes[['Date', 'RaceNumber']].drop_duplicates().values.tolist()
    dict_races = {(race[0], race[1]): [] for race in list_races}

    for row in list_horse_race_outcomes:
        dict_races[(row[1], row[2])].append(row) 

    list_horse_race_outcomes = df_horse_race_outcomes.values.tolist()

    dict_horse_occurrence = defaultdict(float)
    dict_horse_won = defaultdict(float)
    dict_horse_win_ratio = defaultdict(float)
    for horse in horses:
        for opponent in horses:
            if horse != opponent:  
                dict_horse_occurrence[(horse, opponent)] = 0.0
                dict_horse_won[(horse, opponent)] = 0.0
                dict_horse_win_ratio[(horse, opponent)] = 0.0

    dict_possible_points = {}
    total = 0
    for i in range(1, 31):
        total += i 
        dict_possible_points[i] = total - i

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
                    
                    # Measure One
                    # if outcome[3] < outcome_opponent[3]:
                        # dict_horse_occurrence[(outcome[0], outcome_opponent[0])] += 1.0
                        # if outcome[3] < outcome_opponent[3]:                 
                        #     dict_horse_won[(outcome[0], outcome_opponent[0])] += 1.0
                        # wins = dict_horse_won[(outcome[0], outcome_opponent[0])]
                        # occurrences = dict_horse_occurrence[(outcome[0], outcome_opponent[0])]
                        # dict_horse_win_ratio[(outcome[0], outcome_opponent[0])] = wins / occurrences
        
    # print("dict_horse_win_ratio: ", dict_horse_win_ratio)
    # for (horse, opponent), count in dict_horse_win_ratio.items():
    #     print(f"{horse} - {opponent}: {count:.2f}")
    

    df_win_ratio = pd.DataFrame.from_dict(dict_horse_win_ratio, orient='index', columns=['Win Ratio'])
    df_win_ratio.reset_index(inplace=True)
    df_win_ratio[['Horse', 'Opponent']] = pd.DataFrame(df_win_ratio['index'].tolist(), index=df_win_ratio.index)
    df_win_ratio.drop('index', axis=1, inplace=True)
    df_win_ratio.sort_values(by=['Horse', 'Opponent'], ascending=[True, True], inplace=True)

    # for index, row in df_win_ratio.head(1000).iterrows():
    #     print(f"{row['Horse']} vs {row['Opponent']}: {row['Win Ratio']:.2f}")


    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df_win_ratio[['Win Ratio']])
    df_win_ratio_normalized = pd.DataFrame(scaled_values, columns=['Win Ratio Normalized'])
    df_win_ratio = pd.concat([df_win_ratio, df_win_ratio_normalized], axis=1)

    # for index, row in df_win_ratio.head(1000).iterrows():
    #     print(f"{row['Horse']} vs {row['Opponent']}: {row['Win Ratio Normalized']:.2f}")

    print(df_win_ratio.head(20))

    for index, row in df_win_ratio.iterrows():
        key = (row['Horse'], row['Opponent'])
        dict_horse_win_ratio[key] = row['Win Ratio Normalized']


    return dict_horse_win_ratio


def create_race_embeddings():

    dict_win_ratio = create_win_ratio_matrix()

    df = pd.read_csv('./data/Race_comments_gear_horse_competitors_2019_2025.csv')
    df['Placing'] = pd.to_numeric(df['Placing'], errors='coerce')
    df_filtered = df[df['Placing'].notna()]    
    df_filtered['Date'] = pd.to_datetime(df_filtered['Date'])

    horses = df_filtered.sort_values(by=['Horse'], ascending=[True])['Horse'].unique().tolist()

    embedding_dim = 10
    race_embeddings = {
        horse: np.random.randn(embedding_dim) for horse in horses
    }


    learning_rate = 0.1
    num_epochs = 100

    # Gradient descent to update word embeddings
    for epoch in range(num_epochs):
        total_loss = 0
        for (i, j), ratio in dict_win_ratio.items():
            # Calculate dot product of word embeddings
            dot_product = np.dot(race_embeddings[i], race_embeddings[j])
            
            # Calculate difference and update
            diff = dot_product - np.log(ratio)
            total_loss += 0.5 * diff**2
            gradient = diff * race_embeddings[j]
            race_embeddings[i] -= learning_rate * gradient
            
        print(f"Epoch: {epoch+1}, Loss: {total_loss}")

    print("Final horse embeddings:")
    for horse, embedding in race_embeddings.items(): 
        print(f"{horse}: {embedding}")

create_race_embeddings()


# dict_win_ratio = create_win_ratio_matrix()