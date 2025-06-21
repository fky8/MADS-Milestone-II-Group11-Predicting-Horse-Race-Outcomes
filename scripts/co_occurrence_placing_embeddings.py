
# .\.venv\Scripts\Activate.ps1
import numpy as np
from collections import defaultdict
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import os

def get_race_matrix_dataset(start_date='2019-01-01', end_date='2025-12-31') -> pd.DataFrame:   
    """
    gets horse race dataset and applies set filters to build the co-occurrence matrix.
    """

    df = pd.read_csv('./data/Race_comments_gear_horse_competitors_2010_2018.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df2 = pd.read_csv('./data/Race_comments_gear_horse_competitors_2019_2025.csv')    
    df2['Date'] = pd.to_datetime(df2['Date'])
    df_merged = pd.concat([df, df2], ignore_index=True)
    #  filter out horses without a placing due to withdrawal or disqualification
    df_merged['Placing'] = pd.to_numeric(df_merged['Placing'], errors='coerce')
    df_filtered = df_merged[df_merged['Placing'].notna()]    
    

    return df_filtered[(df_filtered['Date'] >= start_date) & (df_filtered['Date'] <= end_date)]


def create_co_occurrence_placing_matrix(target_feature: str ='Jockey',
                            df_race_dataset: pd.DataFrame = None) -> defaultdict[any, float]:
    """
    Create a co-occurrence matrix for how far away the target is from 1st place.
    """

    # The next six lines of code create a dictionary of races 
    # grouped by date and race number to iterate over to calculate
    # the win ratio matrix of the target vs comparison featuires.
    df_target_race_outcomes = df_race_dataset.sort_values(by=['Date', 'RaceNumber'], 
                                ascending=[True, True])\
                                [[target_feature, 'Date', 
                                  'RaceNumber', 'Placing']]
    list_target_race_outcomes = df_target_race_outcomes.values.tolist()

    list_target_races = df_target_race_outcomes[['Date', 'RaceNumber']].\
                            drop_duplicates().values.tolist()
    dict_target_races = {(race[0], race[1]): [] for race in list_target_races}

    for row in list_target_race_outcomes:
        dict_target_races[(row[1], row[2])].append(row)

    # Initialize dictionary to hold occurrences 
    dict_occurrence = defaultdict(float)


    # build the co-occurrence matrix
    count = 1
    for race in list_target_races:        
        for outcome_target in dict_target_races[(race[0], race[1])]:            
            placings = [i for i in range(1, len(dict_target_races[(race[0], race[1])]))] # Assuming placings from 1 to 12
            for placing in placings:                
                distance = abs(placing - outcome_target[3]) # how far away the target is from 1st place                        
                dict_occurrence[(outcome_target[0], placing)] += 1 if distance == 0 else 1.0 / distance
                dict_occurrence[(placing, outcome_target[0])] += 1 if distance == 0 else 1.0 / distance
                                                
    df_occurrence = pd.DataFrame.from_dict(dict_occurrence, orient='index', columns=['Co-occurrence'])
    df_occurrence.reset_index(inplace=True)
    df_occurrence[[target_feature, 'Placing']] = pd.DataFrame(\
        df_occurrence['index'].tolist(), index=df_occurrence.index)
    df_occurrence.drop('index', axis=1, inplace=True)
    df_occurrence.sort_values(by=[target_feature, 'Placing'], 
                             ascending=[True, True], inplace=True)

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df_occurrence[['Co-occurrence']])
    df_occurrence_normalized = pd.DataFrame(scaled_values, columns=['Co-occurrence Normalized'])
    df_occurrence = pd.concat([df_occurrence, df_occurrence_normalized], axis=1)
    
    # if co-occurrence 0, replace with a small value to avoid log division error
    df_occurrence['Co-occurrence'] = df_occurrence['Co-occurrence Normalized'].\
                                            replace(0.0, 0.0001)
    
    for index, row in df_occurrence.iterrows():
        key = (row[target_feature], row['Placing'])
        df_occurrence[key] = row['Co-occurrence Normalized']

    return dict_occurrence


def create_race_embeddings(dict_co_occurrence: defaultdict[any, float] = None)\
                             -> dict[any, np.ndarray]:
    

    targets = set([])
    for (i, j), ratio in dict_co_occurrence.items():
        targets.add(i)
    
    embedding_dim = 50
    race_embeddings = {
        target: np.random.randn(embedding_dim) for target in targets
    }

    learning_rate = 0.01
    num_epochs = 1000
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
        if improvement < 0.0001:
            no_improvement_count += 1
            if no_improvement_count >= 10:
                print(f"No improvement for 10 epochs, stopping early at epoch {epoch+1}, Loss: {total_loss}, \
                        Avg Pair Loss: {avg_pair_loss}, Best Loss: {best_loss}, \
                        Improvement: {prior_best_loss - best_loss}.")
                break
        else:
            no_improvement_count = 0

    return best_embedding


if __name__ == "__main__":

    start_time = datetime.now()
    print("Process started: ", start_time.strftime('%Y-%m-%d %H:%M:%S'))
    number_of_trailing_days = 2147 # 
    # number_of_trailing_days = 7 # 
    trailing_days = 2147
    target_feature = 'Jockey'

    initial_end_date = '2025-05-21'
    df_embeddings = []
    for i in range(0, number_of_trailing_days, 7):
        
        # try:
        end_date = datetime.strptime(initial_end_date, '%Y-%m-%d') - timedelta(days=i)
        start_date = end_date - timedelta(days=trailing_days) 

        df_rd = get_race_matrix_dataset(start_date=str(start_date.strftime('%Y-%m-%d')), 
                                end_date=str(end_date.strftime('%Y-%m-%d')))

        dict_co = create_co_occurrence_placing_matrix(target_feature=target_feature,
                                                    df_race_dataset=df_rd)

        embedding = create_race_embeddings(dict_co_occurrence=dict_co)
        df_embedding = pd.DataFrame.from_dict(embedding, orient='index').reset_index()
        df_embedding.rename(columns={'index': 'Target Feature'}, inplace=True)
        df_embedding['Date End'] = end_date.strftime('%Y-%m-%d')
        df_embedding['Date Begin'] = (end_date - timedelta(days=7)).strftime('%Y-%m-%d')
        df_embeddings.append(df_embedding)

        print(len(embedding))
        # except Exception as e:
        #     print(f"Error processing date {end_date.strftime('%Y-%m-%d')}: {e}")
        #     continue


    df_embeddings_all = pd.concat(df_embeddings, axis=0)

    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    file_path = os.path.join(script_dir, f"../data/" + target_feature + "_Placing_Embeddings_" + str(trailing_days) + ".csv")
    df_embeddings_all.to_csv(file_path, index=False)

    end_time = datetime.now()
    print("Process ended: ", end_time.now().strftime('%Y-%m-%d %H:%M:%S'), "Elapsed Minutes: ", end_time - start_time)

