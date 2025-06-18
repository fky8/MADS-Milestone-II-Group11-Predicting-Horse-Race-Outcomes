import pandas as pd
from typing import List
import sqlite3


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
        
        # 'Placing', 

        'Unnamed: 13', # data is only know during the race, ie. data leakage
        'race_wins', # data is only know during the race, ie. data leakage
        'first_place_win', # data is only know during the race, ie. data leakage
        'second_place_win', # data is only know during the race, ie. data leakage
        'third_place_win', # data is only know during the race, ie. data leakage
        'Race Index', # data is only know during the race, ie. data leakage
        'Race Sub Index', # data is only know during the race, ie. data leakage
        'Rtg.', # data is only know during the race, ie. data leakage
        
        
        'Total Stakes', # useless, doesn't reflect the current state of the horse
        'Age', # useless, doesn't reflect the current state of the horse 
         # (always current age, does not reflect age at time of race)
        'No. of 1-2-3-Starts', # does not reflect the true value at time of race
        'No. of starts in past 10 race meetings', 
        'Current Stable Location (Arrival Date)', 
        'LBW', # distance from winning horse, ie. data leakage
        'Comment', "useless"
        'Last Rating For Retired', "useless"
        'Start of Season Rating', "useless"
        'Current Rating', "useless"
        'Season Stakes', # useless, doesn't reflect the current state of the horse
        'Time 1', # data is only know during the race, ie. data leakage
        'Time 2', # data is only know during the race, ie. data leakage
        'Time 3', # data is only know during the race, ie. data leakage
        'Time 4', # data is only know during the race, ie. data leakage
        'Time 5', # data is only know during the race, ie. data leakage
        'Time 6', # data is only know during the race, ie. data leakage
        'Sectional Time 1', # data is only know during the race, ie. data leakage
        'Sectional Time 2', # data is only know during the race, ie. data leakage
        'Sectional Time 3', # data is only know during the race, ie. data leakage
        'Sectional Time 4', # data is only know during the race, ie. data leakage
        'Sectional Time 5', # data is only know during the race, ie. data leakage
        'Sectional Time 6', # data is only know during the race, ie. data leakage
        'Running Position', # data is only know during the race, ie. data leakage
        'RunningPosition1', # data is only know during the race, ie. data leakage
        'RunningPosition2', # data is only know during the race, ie. data leakage
        'RunningPosition3',# data is only know during the race, ie. data leakage
        'RunningPosition4', # data is only know during the race, ie. data leakage
        'RunningPosition5', # data is only know during the race, ie. data leakage
        'RunningPosition6', # data is only know during the race, ie. data leakage
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
    
    df_merged = df_merged.sort_values(by=[groupby, 'Date'], ascending=[True, True])     

    return df_merged



def merge_horse_jockey_embeddings() -> pd.DataFrame:
    """
    Merges horse jockey co-ocurrence embedding into 
    race data with trailing stats.
    """

    df = get_race_data_with_trailing_stats(groupby='Horse')
    # ['Horse', 'Jockey', 'Date', 'RaceNumber']    
    df_horse_jockey_embeddings = pd.read_csv('./data/Horse_Jockey_Embeddings.csv')
    df_horse_jockey_embeddings.rename(columns={'Date Begin': 'Date_Begin'}, inplace=True)
    df_horse_jockey_embeddings.rename(columns={'Date End': 'Date_End'}, inplace=True)    
    df_horse_jockey_embeddings.rename(columns={'Target Feature': 'Target_Feature'}, inplace=True)    

    embedding_cols = df_horse_jockey_embeddings.columns.tolist()
    embedding_cols = [col for col in embedding_cols if col not in\
                       ['Target_Feature', 'Date_Begin', 'Date_End', 'Trailing Days']]
    
    for i in range(len(embedding_cols)):
        embedding_cols[i] = "H_Emb_" + str(embedding_cols[i]) 
        df_horse_jockey_embeddings.rename(columns={str(i): embedding_cols[i]}, inplace=True)

    # Need an in-memory db connection to crease a between join
    # To perform the same operation in pandas a cross join is required
    # which is requires a lot of memory and is difficult to read.
    # I decided to use sqlite3 to perform the join using SQL syntax.
    conn = sqlite3.connect(':memory:')

    df.to_sql('races', conn, index=False, if_exists='replace')
    df_horse_jockey_embeddings.to_sql('horse_jockey', conn, index=False, if_exists='replace')    

    horse_field_names = ", ".join(embedding_cols)
    print(f"horse_field_names: {horse_field_names}")
    sql_str = """
        SELECT 
            r.*,
            {{horse_field_names}}
        FROM
            races as r
            left join horse_jockey as hj
            on r.Date between hj.Date_Begin and hj.Date_End
            and r.Horse = hj.Target_Feature
    """

    sql_str = sql_str.replace("{{horse_field_names}}", horse_field_names)

    df_merged = pd.read_sql_query(sql_str, conn)

    for i in range(len(embedding_cols)):
        embedding_cols[i] = "J_Emb_" + str(embedding_cols[i]) 
        df_horse_jockey_embeddings.rename(columns={"H_Emb_" + str(i): embedding_cols[i]}, inplace=True)

    df_merged.to_sql('df_merged', conn, index=False, if_exists='replace')
    df_horse_jockey_embeddings.to_sql('horse_jockey', conn, index=False, if_exists='replace')    

    jockey_field_names = ", ".join(embedding_cols)
    print(f"jockey_field_names: {jockey_field_names}")
    
    sql_str = """
        SELECT 
            r.*,
            {{jockey_field_names}} 
        FROM
            df_merged as r
            left join horse_jockey as hj
            on r.Date between hj.Date_Begin and hj.Date_End
            and r.Horse = hj.Target_Feature
    """

    sql_str = sql_str.replace("{{horse_field_names}}", horse_field_names)

    df_merged = pd.read_sql_query(sql_str, conn)

    print(f"df_merged: {len(df_merged)}")

    return df_merged



def get_ml_training_data(list_tr: List[str] ) -> pd.DataFrame:
    """
    Returns a DataFrame with the training data for machine learning models.
    """

    df = get_race_data_with_trailing_stats(groupby='Horse')
    cols = df.columns.tolist()
    trailing_avg_cols = [col for col in cols if col[-3:] in list_tr\
                          and 'Placing' in col\
                          and ('Score range' in col\
                          or 'DistanceMeterAsStr' in col)]

    # !!!!!!!!Need to remove unwanted columns!!!!!!!!!

    return_cols = trailing_avg_cols

    # print(df[return_cols].head(10))
    # print(return_cols)
    # print(f"Total rows in dataset after: {len(df)}")
    return df[return_cols]


# trs = ['TR3']

# trs = ['TR1', 'TR2', 'TR3', 'TR4', 
#       'TR5', 'TR6', 'TR7', 'TR8', 
#       'TR9', 'TR10', 'TR11', 'TR12']

# data = get_ml_training_data(trs)    
# print(data.head(10))


trs = ['TR1', 'TR2', 'TR3', 'TR4', 
      'TR5', 'TR6', 'TR7', 'TR8', 
      'TR9', 'TR10', 'TR11', 'TR12']
df_race = get_ml_training_data(list_tr=trs)

data = merge_horse_jockey_embeddings()
print(len(data))
print(data.head(10))