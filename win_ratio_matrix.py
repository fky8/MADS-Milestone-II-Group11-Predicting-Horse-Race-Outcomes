
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



def create_win_ratio_matrix(target_feature: str ='Horse', 
                            comparison_feature: str ='Jockey', 
                            df_race_dataset: pd.DataFrame = None) -> defaultdict[any, float]:
    """
    Create a win ratio matrix for horses based on horse vs horse.
    """

    list_target = df_race_dataset.sort_values(by=[target_feature], ascending=[True])[target_feature].unique().tolist()
    list_comparison = df_race_dataset.sort_values(by=[comparison_feature], ascending=[True])[comparison_feature].unique().tolist()
    
    horses = df_race_dataset.sort_values(by=['Horse'], ascending=[True])['Horse'].unique().tolist()
     
    
    # The next six lines of code create a dictionary of races 
    # grouped by date and race number to iterate over to calculate
    # the win ratio matrix of the target vs comparison featuires.
    df_target_race_outcomes = df_race_dataset.sort_values(by=['Date', 'RaceNumber'], ascending=[True, True])[[target_feature, comparison_feature, 'Date', 'RaceNumber', 'Placing']]
    list_target_race_outcomes = df_target_race_outcomes.values.tolist()

    list_target_races = df_target_race_outcomes[['Date', 'RaceNumber']].drop_duplicates().values.tolist()
    dict_target_races = {(race[0], race[1]): [] for race in list_target_races}

    for row in list_target_race_outcomes:
        dict_target_races[(row[1], row[2])].append(row)

    # Initialize dictionaries to hold occurrences, wins, and win ratios 
    # for each target_feature and comparison_feature pair
    dict_occurrence = defaultdict(float)
    dict_won = defaultdict(float)
    dict_win_ratio = defaultdict(float)
    for target in list_target:
        for comparison in list_comparison:
            if target != comparison:  
                dict_occurrence[(target, comparison)] = 0.0
                dict_won[(target, comparison)] = 0.0
                dict_win_ratio[(target, comparison)] = 0.0

    # Loop through all races and calculate the win ratios
    # If the target feature is the same as the comparison feature,
    # assume the comparison is an opponent based caclulation.
    # opponent based caclulation the win ratio is calculated 
    # as an average placing difference between the target and comparison features.
    # So if horse A palces 3rd and Horse B places 8th, the points difference is 5,
    # average over all horse A and horse B match-ups. 
    if target_feature == comparison_feature:
        for race in list_target_races:        
            for outcome_target in dict_target_races[(race[0], race[1])]:            
                for outcome_comparison in dict_target_races[(race[0], race[1])]:                
                    if outcome_target[0] != outcome_comparison[1]:                    
                        points = outcome_comparison[4] - outcome_target[4]
                        dict_won[(outcome_target[0], outcome_comparison[1])] += points
                        dict_occurrence[(outcome_target[0], outcome_comparison[1])] += 1.0
                        
                        points = dict_won[(outcome_target[0], outcome_comparison[1])]
                        occurences = dict_occurrence[(outcome_target[0], outcome_comparison[1])]
                        if occurences > 0:
                            dict_win_ratio[(outcome_target[0], outcome_comparison[1])] = points / occurences                 
    else:
    # If the target feature is NOT the same as the comparison feature,
    # assume the comparison is target_feature given the comparison_feature caclulation.
    # Ex. If the target horse is ridden by a given jockey, the calculation will be the 
    # average differnce between 1st place and the horse's finishing placing
        first_place = 1
        for race in list_target_races:        
            for outcome_target in dict_target_races[(race[0], race[1])]:                            
                if outcome_target[0] != outcome_target[1]:                    
                    points = first_place - outcome_target[4]
                    dict_won[(outcome_target[0], outcome_target[1])] += points
                    dict_occurrence[(outcome_target[0], outcome_target[1])] += 1.0
                    
                    points = dict_won[(outcome_target[0], outcome_target[1])]
                    occurences = dict_occurrence[(outcome_target[0], outcome_target[1])]
                    if occurences > 0:
                        dict_win_ratio[(outcome_target[0], outcome_target[1])] = points / occurences


    df_win_ratio = pd.DataFrame.from_dict(dict_win_ratio, orient='index', columns=['Win Ratio'])
    df_win_ratio.reset_index(inplace=True)
    df_win_ratio[[target_feature, comparison_feature]] = pd.DataFrame(df_win_ratio['index'].tolist(), index=df_win_ratio.index)
    df_win_ratio.drop('index', axis=1, inplace=True)
    df_win_ratio.sort_values(by=[target_feature, comparison_feature], ascending=[True, True], inplace=True)

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df_win_ratio[['Win Ratio']])
    df_win_ratio_normalized = pd.DataFrame(scaled_values, columns=['Win Ratio Normalized'])
    df_win_ratio = pd.concat([df_win_ratio, df_win_ratio_normalized], axis=1)
    
    df_win_ratio['Win Ratio Normalized'] = df_win_ratio['Win Ratio Normalized'].replace(0.0, 0.0001)
    
    for index, row in df_win_ratio.iterrows():
        key = (row['Horse'], row['Opponent'])
        dict_win_ratio[key] = row['Win Ratio Normalized']

    return dict_win_ratio



def create_race_embeddings(target_feature: str ='Horse',
                           df_race_dataset: pd.DataFrame = None, 
                           dict_win_ratio: defaultdict[any, float] = None):

    targets = df_race_dataset.sort_values(by=[target_feature], ascending=[True])[target_feature].unique().tolist()

    embedding_dim = 10
    race_embeddings = {
        target: np.random.randn(embedding_dim) for target in targets
    }

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

        print(f"Epoch: {epoch+1}, Loss: {total_loss}, Avg Pair Loss: {total_loss / len(dict_win_ratio)}")

    return race_embeddings


def create_win_ratio_matrix_old(df_race_dataset: 
                                pd.DataFrame =None) -> defaultdict[any, float]:
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

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df_win_ratio[['Win Ratio']])
    df_win_ratio_normalized = pd.DataFrame(scaled_values, columns=['Win Ratio Normalized'])
    df_win_ratio = pd.concat([df_win_ratio, df_win_ratio_normalized], axis=1)
    
    df_win_ratio['Win Ratio Normalized'] = df_win_ratio['Win Ratio Normalized'].replace(0.0, 0.0001)
    
    for index, row in df_win_ratio.iterrows():
        key = (row['Horse'], row['Opponent'])
        dict_horse_win_ratio[key] = row['Win Ratio Normalized']

    return dict_horse_win_ratio



def create_race_embeddings_old(df_race_dataset: pd.DataFrame = None, 
                           dict_win_ratio: defaultdict[any, float] = None):

    horses = df_race_dataset.sort_values(by=['Horse'], ascending=[True])['Horse'].unique().tolist()

    embedding_dim = 10
    race_embeddings = {
        horse: np.random.randn(embedding_dim) for horse in horses
    }

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

        print(f"Epoch: {epoch+1}, Loss: {total_loss}, Avg Pair Loss: {total_loss / len(dict_win_ratio)}")

    return race_embeddings

df_rd = get_race_dataset(start_date='2024-05-01', end_date='2024-5-31')
dict_wr = create_win_ratio_matrix(df_race_dataset=df_rd)
race_embeddings = create_race_embeddings(df_race_dataset=df_rd, dict_win_ratio=dict_wr)



    # dict_possible_points = {}
    # total = 0
    # for i in range(1, 31):
    #     total += i 
    #     dict_possible_points[i] = total - i