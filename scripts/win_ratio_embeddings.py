
# .\.venv\Scripts\Activate.ps1
import numpy as np
from collections import defaultdict
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import os

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
                            comparison_feature: str ='Horse', 
                            df_race_dataset: pd.DataFrame = None) -> defaultdict[any, float]:
    """
    Create a win ratio matrix for horses based on horse vs horse.
    """


    
    # The next six lines of code create a dictionary of races 
    # grouped by date and race number to iterate over to calculate
    # the win ratio matrix of the target vs comparison featuires.
    df_target_race_outcomes = df_race_dataset.sort_values(by=['Date', 'RaceNumber'], 
                                ascending=[True, True])\
                                [[target_feature, comparison_feature, 'Date', 
                                  'RaceNumber', 'Placing']]
    list_target_race_outcomes = df_target_race_outcomes.values.tolist()

    list_target_races = df_target_race_outcomes[['Date', 'RaceNumber']].\
                            drop_duplicates().values.tolist()
    dict_target_races = {(race[0], race[1]): [] for race in list_target_races}

    for row in list_target_race_outcomes:
        dict_target_races[(row[2], row[3])].append(row)

    # Initialize dictionaries to hold occurrences, wins, and win ratios 
    dict_occurrence = defaultdict(float)
    dict_won = defaultdict(float)
    dict_win_ratio = defaultdict(float)

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
                            # dict_win_ratio[(outcome_target[0], 
                            #                 outcome_comparison[1])] = points
                            dict_win_ratio[(outcome_target[0], 
                                            outcome_comparison[1])] = points / occurences
    else:   
    # If the target feature is NOT the same as the comparison feature,
    # assume the comparison is target_feature given the comparison_feature caclulation.
    # Ex. If the target horse is ridden by a given jockey, the calculation will be the 
    # average differnce between 1st place and the horse's finishing placing        
        for race in list_target_races:        
            for outcome_target in dict_target_races[(race[0], race[1])]:
                number_of_placings = len(outcome_target)                         
                if outcome_target[0] != outcome_target[1]:                    

                    points = number_of_placings - outcome_target[4]
                    # from the target perspective to the comparsion feature

                    dict_won[(outcome_target[0], outcome_target[1])] += points                    
                    dict_occurrence[(outcome_target[0], outcome_target[1])] += 1.0                    
                    
                    points = dict_won[(outcome_target[0], outcome_target[1])]
                    occurences = dict_occurrence[(outcome_target[0], outcome_target[1])]
                    if occurences > 0:
                        dict_win_ratio[(outcome_target[0], 
                                        outcome_target[1])] = points / occurences
                                    
                    # from the comparison perspective, the target feature
                    dict_won[(outcome_target[1], outcome_target[0])] += points                                                                
                    dict_occurrence[(outcome_target[1], outcome_target[0])] += 1.0

                    points = dict_won[(outcome_target[1], outcome_target[0])]
                    occurences = dict_occurrence[(outcome_target[1], outcome_target[0])]
                    if occurences > 0:
                        dict_win_ratio[(outcome_target[1],
                                        outcome_target[0])] = points / occurences


    df_win_ratio = pd.DataFrame.from_dict(dict_win_ratio, orient='index', columns=['Win Ratio'])
    df_win_ratio.reset_index(inplace=True)
    comparison_feature_name = 'Opponent' if target_feature == comparison_feature else comparison_feature
    df_win_ratio[[target_feature, comparison_feature_name]] = pd.DataFrame(\
        df_win_ratio['index'].tolist(), index=df_win_ratio.index)
    df_win_ratio.drop('index', axis=1, inplace=True)
    df_win_ratio.sort_values(by=[target_feature, comparison_feature], 
                             ascending=[True, True], inplace=True)

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df_win_ratio[['Win Ratio']])
    df_win_ratio_normalized = pd.DataFrame(scaled_values, columns=['Win Ratio Normalized'])
    df_win_ratio = pd.concat([df_win_ratio, df_win_ratio_normalized], axis=1)
    
    df_win_ratio['Win Ratio Normalized'] = df_win_ratio['Win Ratio Normalized'].\
                                            replace(0.0, 0.0001)
    
    for index, row in df_win_ratio.iterrows():
        key = (row[target_feature], row[comparison_feature_name])
        dict_win_ratio[key] = row['Win Ratio Normalized']

    return dict_win_ratio

def create_race_embeddings(dict_win_ratio: defaultdict[any, float] = None)\
                             -> dict[any, np.ndarray]:
    
    targets = set([])
    for (i, j), ratio in dict_win_ratio.items():
        targets.add(i)

    embedding_dim = 5
    race_embeddings = {
        target: np.random.randn(embedding_dim) for target in targets
    }

    learning_rate = 0.02
    num_epochs = 1000
    best_loss = float('inf')
    best_embedding = {}
    no_improvement_count = 0
    for epoch in range(num_epochs):
        total_loss = 0
        for (i, j), ratio in dict_win_ratio.items():
            dot_product = np.dot(race_embeddings[i], race_embeddings[j])
            diff = dot_product - np.log(ratio)
            total_loss += 0.5 * diff**2
            gradient = diff * race_embeddings[j]
            race_embeddings[i] -= learning_rate * gradient

        avg_pair_loss = total_loss / len(dict_win_ratio)
    
        prior_best_loss = best_loss
        if best_loss > avg_pair_loss:
            best_loss = avg_pair_loss
            best_embedding = race_embeddings.copy()

        improvement = prior_best_loss - best_loss
        if improvement < 0.0001:
            no_improvement_count += 1
            if no_improvement_count >= 10:
                print(f"No improvement for 10 epochs, stopping early at epoch {epoch+1}, Loss: {total_loss}, \
                        Avg Pair Loss: {avg_pair_loss}, Best Loss: {best_loss}, \
                        Improvement: {prior_best_loss - best_loss}.")
                break
        else:
            no_improvement_count = 0

        # print(f"Epoch: {epoch+1}, Loss: {total_loss}, \
        #       Avg Pair Loss: {avg_pair_loss}, Best Loss: {best_loss}, \
        #       Improvement: {prior_best_loss - best_loss}")

    return best_embedding




start_time = datetime.now()
print("Process started: ", start_time.strftime('%Y-%m-%d %H:%M:%S'))
number_of_trailing_days = 1967 # last two years of data
# number_of_trailing_days = 7 # 
trailing_days = 180
target_feature = 'Horse'
comparison_feature = 'Jockey'

initial_end_date = '2025-05-21'
df_embeddings = []
for i in range(0, number_of_trailing_days, 7):
    
    try:
        end_date = datetime.strptime(initial_end_date, '%Y-%m-%d') - timedelta(days=i)
        start_date = end_date - timedelta(days=trailing_days) 

        df_rd = get_race_dataset(start_date=str(start_date.strftime('%Y-%m-%d')), 
                                end_date=str(end_date.strftime('%Y-%m-%d')))

        dict_wr = create_win_ratio_matrix(target_feature=target_feature, 
                                        comparison_feature=comparison_feature, 
                                        df_race_dataset=df_rd)

        embedding = create_race_embeddings(dict_win_ratio=dict_wr)
        # df_embedding = pd.DataFrame(list(embedding.items()), columns=['Target Feature', 'Embedding'])
        df_embedding = pd.DataFrame.from_dict(embedding, orient='index').reset_index()
        df_embedding.rename(columns={'index': 'Target Feature'}, inplace=True)
        df_embedding['Date Begin'] = end_date.strftime('%Y-%m-%d')
        df_embedding['Date End'] = (end_date - timedelta(days=trailing_days)).strftime('%Y-%m-%d')
        df_embedding['Trailing Days'] = trailing_days
        df_embeddings.append(df_embedding)
        # print(df_embedding.head(10))
        # print(embedding)
        print(len(embedding))
    except Exception as e:
        print(f"Error processing date {end_date.strftime('%Y-%m-%d')}: {e}")
        continue


df_embeddings_all = pd.concat(df_embeddings, axis=0)

script_dir = os.path.dirname(os.path.abspath(__file__)) 
file_path = os.path.join(script_dir, f"../data/" + target_feature + "_" + comparison_feature +"_Embeddings.csv")
df_embeddings_all.to_csv(file_path, index=False)

end_time = datetime.now()
print("Process ended: ", end_time.now().strftime('%Y-%m-%d %H:%M:%S'), "Elapsed Minutes: ", end_time - start_time)


# I need horse transformed names for joiner, due to repeat names in the dataset.
# I need to create an peak trailing indicator for horse performance. 
# days_from_peak = race_date - date(by="track_meters", max(peak_meters_per_second))
# create a plot to see if there is a realationship between time and peak performance.
# 





    # dict_possible_points = {}
    # total = 0
    # for i in range(1, 31):
    #     total += i 
    #     dict_possible_points[i] = total - i