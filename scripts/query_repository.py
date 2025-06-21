import pandas as pd
from typing import List
import os


def get_race_dataset_old(start_date='2019-01-01', end_date='2025-12-31'):   
    """
    gets horse race dataset and applies set filters to standardize the race data
    accross functions.  
    """

    df = pd.read_csv('./data/Race_comments_gear_horse_competitors_2019_2025.csv')
    #  filter out horses without a placing due to withdrawal or disqualification
    df['Placing'] = pd.to_numeric(df['Placing'], errors='coerce')
    df_filtered = df[df['Placing'].notna()]    
    df_filtered['Date'] = pd.to_datetime(df_filtered['Date'])

    df_filtered['DistanceMeterAsStr'] = df_filtered['DistanceMeter'].astype(str)
    df_filtered['Act. Wt.'] = df_filtered['Act. Wt.'].astype(float)
    df_filtered['Declar. Horse Wt.'] = df_filtered['Declar. Horse Wt.'].astype(float)

    return df_filtered[(df_filtered['Date'] >= start_date) & (df_filtered['Date'] <= end_date)]


def get_race_dataset(start_date='2019-01-01', end_date='2025-12-31'):   
    """
    gets horse race dataset and applies set filters to standardize the race data
    accross functions.  
    """

    df = pd.read_csv('./data/Race_comments_gear_horse_competitors_2019_2025.csv')
    
    
    # Cleanse Data
    #  filter out horses without a placing due to withdrawal or disqualification
    df['Placing'] = pd.to_numeric(df['Placing'], errors='coerce')
    df_filtered = df[df['Placing'].notna()]    
    df_filtered['Date'] = pd.to_datetime(df_filtered['Date'])

    df_filtered['DistanceMeterAsStr'] = df_filtered['DistanceMeter'].astype(str)
    df_filtered['Act. Wt.'] = df_filtered['Act. Wt.'].astype(float)
    df_filtered['Declar. Horse Wt.'] = df_filtered['Declar. Horse Wt.'].astype(float)

    # Drop fileds that are associated with dataleakage 
    drop_cols = [
        

        
        'Total Stakes',
        'Age',        
        'No. of 1-2-3-Starts',
        'No. of starts in past 10 race meetings', 
        'Current Stable Location (Arrival Date)', 
        'LBW',
        'Comment',
        'Last Rating For Retired',
        'Start of Season Rating',
        'Current Rating',
        'Season Stakes',
        'Time 1',
        'Time 2',
        'Time 3',
        'Time 4',
        'Time 5',
        'Time 6',
        'Sectional Time 1',
        'Sectional Time 2',
        'Sectional Time 3',
        'Sectional Time 4',
        'Sectional Time 5',
        'Sectional Time 6',
        'Running Position',
        'RunningPosition1',
        'RunningPosition2',
        'RunningPosition3',
        'RunningPosition4',
        'RunningPosition5',
        'RunningPosition6'
    ]

    df_filtered.drop(columns=drop_cols, inplace=True)

    return df_filtered[(df_filtered['Date'] >= start_date) & (df_filtered['Date'] <= end_date)]

def get_trailing_average_win_stats(groupby: str ='Horse', 
                                   metric_grouping: str ='Score range')\
                                   -> pd.DataFrame:
    """
    Pre calculated trailing average win stats by:
    groupby: 'Horse', 'Jockey'
    metric_grouping: 'Score range', 'Going', 'DistanceMeterAsStr', Dr.
    
    """

    try:
        df = pd.read_csv(f"./data/{groupby}_trailing_stats_{metric_grouping}.csv")
        df['Date'] = pd.to_datetime(df['Date'])
    except FileNotFoundError:
        print("File not found. Please ensure the dataset is available at the specified path.")
        return pd.DataFrame()
    
    return df


def get_trailing_placing_averages(groupby: str ='Horse') -> pd.DataFrame:
    """
    Pre calculated trailing average win stats by:
    groupby: 'Horse', 'Jockey'
    metric_grouping: 'Score range', 'Going', 'DistanceMeterAsStr', Dr.
    
    """

    try:
        df = pd.read_csv(f"./data/{groupby}_trailing_placing_averages.csv")
        df['Date'] = pd.to_datetime(df['Date'])
    except FileNotFoundError:
        print("File not found. Please ensure the dataset is available at the specified path.")
        return pd.DataFrame()

    return df

def get_race_data_with_trailing_stats(groupby: str ='Horse') -> pd.DataFrame:
    """
    """
    df_trailing_score_range = get_trailing_average_win_stats(groupby='Horse',
                                                      metric_grouping='Score range')

    df_trailing_draw = get_trailing_average_win_stats(groupby='Horse',
                                                      metric_grouping='Dr.')
    df_trailing_draw['Dr.'] == df_trailing_draw['Dr.'].astype(float)
    df_trailing_going = get_trailing_average_win_stats(groupby='Horse',
                                                      metric_grouping='Going')
    df_trailing_distance = get_trailing_average_win_stats(groupby='Horse',
                                                      metric_grouping='DistanceMeterAsStr')    

     

    df = get_race_dataset()
    df['Dr.'] = df['Dr.'].astype(float)
    df['DistanceMeterAsStr'] = df['DistanceMeter'].astype(float)
    print(f"Total rows in dataset before: {len(df)}")

    df_merged = pd.merge(df, df_trailing_score_range, on=[groupby, 'Score range', 'Date'], how='left', suffixes=('', '_y'))
    drop_cols = [col for col in df_merged.columns if col[-2:] == '_y']
    df_merged = df_merged.drop(columns=drop_cols, errors='ignore')

    df_merged = pd.merge(df_merged, df_trailing_draw, on=[groupby, 'Dr.', 'Date'], how='left', suffixes=('', '_y'))
    drop_cols = [col for col in df_merged.columns if col[-2:] == '_y']
    df_merged = df_merged.drop(columns=drop_cols, errors='ignore')
    
    
    df_merged = pd.merge(df_merged, df_trailing_going, on=[groupby, 'Going', 'Date'], how='left', suffixes=('', '_y'))
    drop_cols = [col for col in df_merged.columns if col[-2:] == '_y']
    df_merged = df_merged.drop(columns=drop_cols, errors='ignore')
    df_merged = pd.merge(df_merged, df_trailing_distance, on=[groupby, 'DistanceMeterAsStr', 'Date'], how='left', suffixes=('', '_y'))            
    drop_cols = [col for col in df_merged.columns if col[-2:] == '_y']
    df_merged = df_merged.drop(columns=drop_cols, errors='ignore')


    df_trailing_placing_averages = get_trailing_placing_averages(groupby=groupby)
    df_merged = pd.merge(df_merged, df_trailing_placing_averages, on=[groupby, 'Date'], how='left', suffixes=('', '_z'))            
    drop_cols = ['Date_z', 'Horse_z', 'Jockey_z', 'Placing_z']
    df_merged = df_merged.drop(columns=drop_cols, errors='ignore')
    

    df_merged = df_merged.sort_values(by=[groupby, 'Date'], ascending=[True, True])     

    return df_merged

def merge_horse_jockey_embeddings(df_trailing_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Merges horse jockey co-ocurrence embedding into 
    race data with trailing stats.
    """
    # ['Horse', 'Jockey', 'Date', 'RaceNumber']    
    df_horse_jockey_embeddings = pd.read_csv('./data/Jockey_Jockey_Embeddings_540_flat.csv')
    df_horse_jockey_embeddings['Date'] = pd.to_datetime(df_horse_jockey_embeddings['Date'])
    df_horse_jockey_embeddings.rename(columns={'Target Feature': 'Target_Feature'}, inplace=True)    

    df_merged = pd.merge(df_trailing_stats, df_horse_jockey_embeddings,
                         left_on=['Jockey', 'Date'],
                         right_on=['Target_Feature', 'Date'],
                         how='left',
                         suffixes=('', ''))

    df_merged.drop(columns=['Target_Feature_y', 
                            'Target_Feature_z',
                            'Date_y',
                            'Date_z'], 
                            inplace=True, errors='ignore')

    print(f"df_merged: {len(df_merged)}")

    return df_merged


def merge_jockey_jockey_embeddings(df_trailing_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Merges horse jockey co-ocurrence embedding into 
    race data with trailing stats.
    """

    df_trailing_stats = get_race_data_with_trailing_stats(groupby='Horse')
    # ['Horse', 'Jockey', 'Date', 'RaceNumber']    
    df_horse_jockey_embeddings = pd.read_csv('./data/Jockey_Jockey_Embeddings_540_flat.csv')
    # df_horse_jockey_embeddings['Date'] = pd.to_datetime(df_horse_jockey_embeddings['Date'])
    df_horse_jockey_embeddings.rename(columns={'Target Feature': 'Target_Feature'}, inplace=True)    


    embedding_cols = df_horse_jockey_embeddings.columns.tolist()
    embedding_cols = [col for col in embedding_cols if col not in\
                       ['Target_Feature', 'Date', 'Date Begin', 'Date End']]
    
    for i in range(len(embedding_cols)):
        embedding_cols[i] = "J_Emb_" + str(embedding_cols[i]) 
        df_horse_jockey_embeddings.rename(columns={str(i): embedding_cols[i]}, inplace=True)


    df_merged = pd.merge(df_trailing_stats, df_horse_jockey_embeddings,
                         left_on=['Jockey'],
                         right_on=['Target_Feature'],
                         how='left',
                         suffixes=('', '_z'))

    df_merged.drop(columns=['Target_Feature_y', 
                            'Target_Feature_z',
                            'Date_y',
                            'Date_z'], 
                            inplace=True, errors='ignore')

    print(f"df_merged: {len(df_merged)}")

    return df_merged


def merge_horse_embeddings(df_trailing_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Merges horse jockey co-ocurrence embedding into 
    race data with trailing stats.
    """


    # ['Horse', 'Jockey', 'Date', 'RaceNumber']    
    df_horse_embeddings = pd.read_csv('./data/Horse_Horse_Embeddings_180_flat.csv')
    df_horse_embeddings['Date'] = pd.to_datetime(df_horse_embeddings['Date'])
    df_horse_embeddings.rename(columns={'Target Feature': 'Target_Feature'}, inplace=True)    


    df_merged = pd.merge(df_trailing_stats, df_horse_embeddings,
                         left_on=['Horse', 'Date'],
                         right_on=['Target_Feature', 'Date'],
                         how='left',
                         suffixes=('', '_z'))

    df_merged.drop(columns=['Target_Feature_y', 
                            'Target_Feature_z',
                            'Date_y',
                            'Date_z'], 
                            inplace=True, errors='ignore')

    return df_merged





def build_ml_training_data():
    """
    Returns a DataFrame with the training data for machine learning models.
    """

    df_race = get_race_dataset(start_date='2019-01-01', end_date='2025-12-31')
    original_cols = df_race.columns.tolist()

    
    df = get_race_data_with_trailing_stats(groupby='Horse')


    list_tr = ['TR1','TR2','TR3','TR4','TR5']
    list_tr2 = ['Placing_TR1','Placing_TR2','Placing_TR3','Placing_TR4',
                'Placing_TR5','Placing_TR6','Placing_TR7','Placing_TR8',''
                'Placing_TR9','Placing_TR10'] 
    
    cols = df.columns.tolist()
    trailing_avg_cols = [col for col in cols if col[-3:] in list_tr\
                          and 'Placing' in col\
                          and ('Score range' in col\
                          or 'DistanceMeterAsStr' in col)]

    trailing_avg_cols2 = [col for col in cols if col in list_tr2]


    df_merged = merge_horse_jockey_embeddings(df)
    # df_merged = merge_jockey_jockey_embeddings(df)

    df_merged = merge_horse_embeddings(df_merged)

    cols = df_merged.columns.tolist()
    embedding_cols = [col for col in cols if col[0:6] in ['H_Emb_', 'J_Emb_']]

    return_cols = original_cols + trailing_avg_cols + trailing_avg_cols2 + embedding_cols
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    file_path = os.path.join(script_dir, f"../data/ml_dataset_2019_2025.csv")
    df_merged[return_cols].to_csv(file_path, index=False)


def get_ml_training_data() -> pd.DataFrame:
    return pd.read_csv('./data/ml_dataset_2019_2025.csv')



if __name__ == "__main__":
    build_ml_training_data()


# trs = ['TR3']

# trs = ['TR1', 'TR2', 'TR3', 'TR4', 
#       'TR5', 'TR6', 'TR7', 'TR8', 
#       'TR9', 'TR10', 'TR11', 'TR12']

# data = get_ml_training_data(trs)    
# print(data.head(10))


# trs = ['TR1', 'TR2', 'TR3', 'TR4', 
#       'TR5', 'TR6', 'TR7', 'TR8', 
#       'TR9', 'TR10', 'TR11', 'TR12']
# df_race = get_ml_training_data(list_tr=trs)

# data = merge_horse_jockey_embeddings()
# print(len(data))
# print(data.head(10))






        # 'Unnamed: 13', # data is only know during the race, ie. data leakage
        # 'race_wins', # data is only know during the race, ie. data leakage
        # 'first_place_win', # data is only know during the race, ie. data leakage
        # 'second_place_win', # data is only know during the race, ie. data leakage
        # 'third_place_win', # data is only know during the race, ie. data leakage
        # 'Race Index', # data is only know during the race, ie. data leakage
        # 'Race Sub Index', # data is only know during the race, ie. data leakage
        # 'Rtg.', # data is only know during the race, ie. data leakage
        
        
        # 'Total Stakes', # useless, doesn't reflect the current state of the horse
        # 'Age', # useless, doesn't reflect the current state of the horse 
        #  # (always current age, does not reflect age at time of race)
        # 'No. of 1-2-3-Starts', # does not reflect the true value at time of race
        # 'No. of starts in past 10 race meetings', 
        # 'Current Stable Location (Arrival Date)', 
        # 'LBW', # distance from winning horse, ie. data leakage
        # 'Comment', "useless"
        # 'Last Rating For Retired', "useless"
        # 'Start of Season Rating', "useless"
        # 'Current Rating', "useless"
        # 'Season Stakes', # useless, doesn't reflect the current state of the horse
        # 'Time 1', # data is only know during the race, ie. data leakage
        # 'Time 2', # data is only know during the race, ie. data leakage
        # 'Time 3', # data is only know during the race, ie. data leakage
        # 'Time 4', # data is only know during the race, ie. data leakage
        # 'Time 5', # data is only know during the race, ie. data leakage
        # 'Time 6', # data is only know during the race, ie. data leakage
        # 'Sectional Time 1', # data is only know during the race, ie. data leakage
        # 'Sectional Time 2', # data is only know during the race, ie. data leakage
        # 'Sectional Time 3', # data is only know during the race, ie. data leakage
        # 'Sectional Time 4', # data is only know during the race, ie. data leakage
        # 'Sectional Time 5', # data is only know during the race, ie. data leakage
        # 'Sectional Time 6', # data is only know during the race, ie. data leakage
        # 'Running Position', # data is only know during the race, ie. data leakage
        # 'RunningPosition1', # data is only know during the race, ie. data leakage
        # 'RunningPosition2', # data is only know during the race, ie. data leakage
        # 'RunningPosition3',# data is only know during the race, ie. data leakage
        # 'RunningPosition4', # data is only know during the race, ie. data leakage
        # 'RunningPosition5', # data is only know during the race, ie. data leakage
        # 'RunningPosition6', # data is only know during the race, ie. data leakage