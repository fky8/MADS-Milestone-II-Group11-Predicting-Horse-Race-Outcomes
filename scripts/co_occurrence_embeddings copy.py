
# .\.venv\Scripts\Activate.ps1
import numpy as np
from collections import defaultdict
from numpy._core.numeric import nan
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import math
import multiprocessing
import time


current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.dirname(current_dir)
sys.path.append(scripts_dir)

from scripts.query_repository import get_trailing_placing_averages

def get_race_matrix_dataset(start_date='2019-01-01', end_date='2025-12-31') -> pd.DataFrame:   
    """
    gets horse race dataset and applies set filters to build the co-occurrence matrix.
    """

    df = pd.read_csv('./data/Race_comments_gear_horse_competitors_2010_2018.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df2 = pd.read_csv('./data/Race_comments_gear_horse_competitors_2019_2025.csv')    
    df2['Date'] = pd.to_datetime(df2['Date'])
    df_merged = pd.concat([df, df2], ignore_index=True)
    
    df_trailing_horse_placing_averages = get_trailing_placing_averages(groupby='Horse')
    df_merged = pd.merge(df_merged, df_trailing_horse_placing_averages, 
                         on=['Horse', 'Date'], how='left', suffixes=('', '_z'))            
    drop_cols = ['Date_z', 'Horse_z', 'Jockey_z', 'Placing_z']
    df_merged = df_merged.drop(columns=drop_cols, errors='ignore')
    
    #  filter out horses without a placing due to withdrawal or disqualification
    df_merged['Placing'] = pd.to_numeric(df_merged['Placing'], errors='coerce')
    df_filtered = df_merged[df_merged['Placing'].notna()]    
    

    return df_filtered[(df_filtered['Date'] >= start_date) & (df_filtered['Date'] <= end_date)]


def create_co_occurrence_matrix(target_feature: str ='Horse', 
                                comparison_feature: str ='Horse',
                                placing_feature: str = 'Placing',
                                df_race_dataset: pd.DataFrame = None) -> defaultdict[any, float]:
    """
    Create a placing co-occurrence matix for horses based on comparison vs target.
    Ex. Horse vs Horse, Jockey vs Jockey, etc.
    """

    # The next six lines of code create a dictionary of races 
    # grouped by date and race number to iterate over to calculate
    # the win ratio matrix of the target vs comparison featuires.
    df_target_race_outcomes = df_race_dataset.sort_values(by=['Date', 'RaceNumber'], 
                                ascending=[True, True])\
                                [[target_feature, comparison_feature, 'Date', 
                                  'RaceNumber', placing_feature, 'Placing']]

    # print('placing_feature', placing_feature)
    is_trailing_average = True if 'Placing_TR' in placing_feature else False
    tr = int(placing_feature.strip('Placing_TR')) if is_trailing_average else 0

    print('is_trailing_average', is_trailing_average, str(tr))

    list_target_race_outcomes = df_target_race_outcomes.values.tolist()

    list_target_races = df_target_race_outcomes[['Date', 'RaceNumber']].\
                            drop_duplicates().values.tolist()
    dict_target_races = {(race[0], race[1]): [] for race in list_target_races}

    for row in list_target_race_outcomes:
        dict_target_races[(row[2], row[3])].append(row)


    dict_co_occurrences = defaultdict(float)

    # Loop through all races and calculate the win ratios
    # If the target feature is the same as the comparison feature,
    # assume the comparison is an opponent based caclulation ie. Horse vs Horse.
    # Calculate the distance between the target and comparison features
    # So if horse A palces 3rd and Horse B places 8th, the points difference is 5,
    # average over all horse A and horse B match-ups. 
    if target_feature == comparison_feature:
        for race in list_target_races:        
            for target in dict_target_races[(race[0], race[1])]:            
                for comparison in dict_target_races[(race[0], race[1])]:                
                    if target[0] != comparison[1]:
                                            
                        if is_trailing_average:
                            # calculate the average placing for the trailing days plus 
                            # the current placing for the comparison and target
                            c_plc = comparison[5]
                            c_avg = comparison[4]
                            comp = (c_avg+c_plc)/2.0 if math.isnan(c_avg) == False and c_avg != 0 else c_plc                        
                            t_plc = target[5]
                            t_avg = target[4]
                            tgt= (t_avg+t_plc)/2.0 if math.isnan(t_avg) == False and t_avg != 0 else t_plc
                        else:
                            comp = comparison[5]
                            tgt = target[5]

                        dist = abs(tgt - comp) # how far away the target is from the comparison
                        if dist == 0:
                            dist = 1 # handle div zero
                        # calculate the co-occurrence ratio                          
                        dict_co_occurrences[(target[0], comparison[1])] += (1.0 / dist)
    
    return dict_co_occurrences



def create_embeddings(dict_co_occurrence: defaultdict[any, float] = None,
                      dims: int = 50, 
                      l_rate: float = 0.01, 
                      epochs: int = 1000
                      ) -> dict[any, np.ndarray]:
    """
    Creates embeddings resembling GloVe based on the co-occurrence matrix.
    """
    
    targets = set([])
    for (i, j), ratio in dict_co_occurrence.items():
        targets.add(i)

    embedding_dim = dims
    race_embeddings = {
        target: np.random.randn(embedding_dim) for target in targets
    }

    learning_rate = l_rate
    num_epochs = epochs
    best_loss = float('inf')
    best_embedding = {}
    no_improvement_count = 0
    for epoch in range(num_epochs):
        total_loss = 0
        for (i, j), ratio in dict_co_occurrence.items():
            dot_product = np.dot(race_embeddings[i], race_embeddings[j])
            diff = dot_product - np.log(ratio)
            total_loss += 0.5 * diff**2
            gradient = diff * race_embeddings[j]
            race_embeddings[i] -= learning_rate * gradient

        avg_pair_loss = total_loss / len(dict_co_occurrence)
    
        prior_best_loss = best_loss
        if best_loss > avg_pair_loss:
            best_loss = avg_pair_loss
            best_embedding = race_embeddings.copy()

        improvement = prior_best_loss - best_loss
        # if epoch % 20 == 0:
        #     print(f"epoch {epoch+1}, Loss: {total_loss}, Avg Pair Loss: {avg_pair_loss},  \
        #             Best Loss: {best_loss}, Improvement: {prior_best_loss - best_loss}.")

        if improvement < 0.0001:
            no_improvement_count += 1
            if no_improvement_count >= 10:
                print(f"No improvement for 10 epochs, stopping early at epoch {epoch+1}, \
                      Loss: {total_loss}, Avg Pair Loss: {avg_pair_loss},  \
                      Best Loss: {best_loss}, Improvement: {prior_best_loss - best_loss}.")
                break
        else:
            no_improvement_count = 0

    return best_embedding


    # number_of_trailing_days = 7 # 
def create_embeddings_sets(end_date: str = '2025-05-21',
                           past_days: int = 1967, 
                           trailing_days: int = 180,
                           interval: int = 7,
                           target_feature: str = 'Horse',
                           comparison_feature: str = 'Horse',
                           placing_feature: str = 'Placing',                           
                           dims: int = 50, 
                           l_rate: float = 0.01, 
                           epochs: int = 1000                                                      
                           ) -> None:
    """
    Creates a new set of embeddings and save them to a csv file based on 
    the parameters: 
    
    end_date:   The initial end date to start the embedding calculations.
    
    past_days:  The number of backwards looking days for weekly calculations.
                Ex. end_Date = 3/29/2025 and past_days = 28 days then 4
                embeddings are created for the weeks of 3/29, 3/22, 3/15,
                and 3/8.
    
    trailing_days: The trailing days used to create the embedding.
                    Ex. trailing_days = 180 days means the co-occurrence matrix
                    is created from the trailing 180 days from the end_date.

    interval: The number of days to step backwards to create the 
                         next embedding.
                         Ex. if prediction_interval = 7 days then a new embedding
                         is created every 7 days (for every week) until the 
                         past_days is reached.

    placing_feature: Overrides the default placing feature to use for the 
                    co-occurrence matrix.    
                    EX. if placing_feautre = 'Placing_TR2' then the co-occurrence 
                    matrix is calculated using a horse's trailing 2nd placing average.

    comparison_feature: The comparison feature to use for the co-occurrence matrix.

    target_feature: The target feature to use for the co-occurrence matrix.

    Ex. if comparison_feature = 'Horse' and target_feature = 'Horse'
        then the co-occurrence matrix is calculated using the comparison horse's placing 
        verses the target horse's placing.


    """

    start_time = datetime.now()
    print("Process started: ", start_time.strftime('%Y-%m-%d %H:%M:%S'))
    initial_end_date = end_date
    df_embeddings = []
    for i in range(0, past_days, interval):
        try:
            end_date = datetime.strptime(initial_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_date = end_date - timedelta(days=trailing_days) 

            # The effective dates that the embeddings are valid for to avoid data leakage.
            # Ex. if end_date = 2025-05-21 and past_days = 1967 then the effective_end_date is
            effective_end_date = end_date + timedelta(days=interval)
            effective_start_date = end_date

            df_rd = get_race_matrix_dataset(start_date=str(start_date.strftime('%Y-%m-%d')), 
                                            end_date=str(end_date.strftime('%Y-%m-%d')))
            print("finished co-occurrence matrix for date: ", start_time.strftime('%Y-%m-%d %H:%M:%S'))
            dict_co = create_co_occurrence_matrix(target_feature=target_feature, 
                                                comparison_feature=comparison_feature, 
                                                df_race_dataset=df_rd,
                                                placing_feature=placing_feature)

            embedding = create_embeddings(dict_co_occurrence=dict_co,
                                        dims=dims, 
                                        l_rate=l_rate, 
                                        epochs=epochs)
            
            df_embedding = pd.DataFrame.from_dict(embedding, orient='index').reset_index()
            df_embedding.rename(columns={'index': 'Target Feature'}, inplace=True)
            df_embedding['Date End'] = effective_end_date.strftime('%Y-%m-%d')
            df_embedding['Date Begin'] = effective_start_date.strftime('%Y-%m-%d')
            df_embeddings.append(df_embedding)
            print(len(embedding))
        except Exception as e:
            print(f"Error processing date {end_date.strftime('%Y-%m-%d')}: {e}")
            continue

    df_embeddings_all = pd.concat(df_embeddings, axis=0)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = f"../data/{target_feature}_{comparison_feature}_Embeddings_{str(trailing_days)}.csv"
    file_path_joined = os.path.join(script_dir, file_path)
    df_embeddings_all.to_csv(file_path_joined, index=False)

    end_time = datetime.now()
    print("Process ended: ", end_time.now().strftime('%Y-%m-%d %H:%M:%S'), "Elapsed Minutes: ", end_time - start_time)


if __name__ == "__main__":
    # 1967


    process1 = multiprocessing.Process(target=create_embeddings_sets(end_date = '2025-05-21',
                                        past_days = 365 * 7, # 7 years of weekly embeddings
                                        trailing_days = 365 * 7,
                                        interval = 7 * 8, # 8 weeks of embeddings
                                        target_feature = 'Horse',
                                        comparison_feature = 'Horse',
                                        placing_feature = 'Placing',
                                        dims = 50, 
                                        l_rate = 0.01, 
                                        epochs = 1000,
                                        file_suffix="2025-05-21"                            
                                        ),args=("One",))
    process2 = multiprocessing.Process(target=worker_function, args=("Two",))





