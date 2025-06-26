from datetime import datetime, timedelta
import pandas as pd
from typing import List
import os
import sqlite3
import csv

def get_race_dataset(start_date='2019-01-01', end_date='2025-12-31'):   
    """
    gets horse race dataset and applies set filters to standardize the race data
    accross functions.  
    """

    df = pd.read_csv('./data/Race_comments_gear_horse_competitors_2019_2025.csv')
    #  Cleanse Data
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
    groupby: 'Horse'
    metric_grouping: 'Score range', 'Going', 'DistanceMeterAsStr', Dr.    
    """

    try:
        df = pd.read_csv(f"./data/{groupby}_trailing_stats_{metric_grouping}.csv")
        df['Date'] = pd.to_datetime(df['Date'])
    except FileNotFoundError:
        print("File not found.")
        return pd.DataFrame()
    
    return df

def get_trailing_placing_averages(groupby: str ='Horse') -> pd.DataFrame:
    """
    Pre calculated trailing average win stats by:
    groupby: 'Horse'    
    """

    try:
        df = pd.read_csv(f"./data/{groupby}_trailing_placing_averages.csv")
        df['Date'] = pd.to_datetime(df['Date'])
    except FileNotFoundError:
        print("File not found.")
        return pd.DataFrame()

    return df

def get_race_data_with_trailing_stats(groupby: str ='Horse') -> pd.DataFrame:
    """
    Merges all trailing average stats
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

    df_merged = pd.merge(df, df_trailing_score_range, 
                         on=[groupby, 'Score range', 'Date'], 
                         how='left', suffixes=('', '_y'))
    drop_cols = [col for col in df_merged.columns if col[-2:] == '_y']
    df_merged = df_merged.drop(columns=drop_cols, errors='ignore')

    df_merged = pd.merge(df_merged, df_trailing_draw, 
                         on=[groupby, 'Dr.', 'Date'], 
                         how='left', suffixes=('', '_y'))
    drop_cols = [col for col in df_merged.columns if col[-2:] == '_y']
    df_merged = df_merged.drop(columns=drop_cols, errors='ignore')
    
    df_merged = pd.merge(df_merged, df_trailing_going, 
                         on=[groupby, 'Going', 'Date'], 
                         how='left', suffixes=('', '_y'))
    drop_cols = [col for col in df_merged.columns if col[-2:] == '_y']
    df_merged = df_merged.drop(columns=drop_cols, errors='ignore')

    df_merged = pd.merge(df_merged, df_trailing_distance, 
                         on=[groupby, 'DistanceMeterAsStr', 'Date'], 
                         how='left', suffixes=('', '_y'))            
    drop_cols = [col for col in df_merged.columns if col[-2:] == '_y']
    df_merged = df_merged.drop(columns=drop_cols, errors='ignore')

    df_trailing_placing_averages = get_trailing_placing_averages(groupby=groupby)
    df_merged = pd.merge(df_merged, df_trailing_placing_averages, 
                         on=[groupby, 'Date'], 
                         how='left', suffixes=('', '_z'))            
    drop_cols = ['Date_z', 'Horse_z', 'Jockey_z', 'Placing_z']
    df_merged = df_merged.drop(columns=drop_cols, errors='ignore')
    
    df_merged = df_merged.sort_values(by=[groupby, 'Date'], 
                                      ascending=[True, True])     

    return df_merged

# df_horse_jockey_embeddings = pd.read_csv('./data/Jockey_Jockey_Embeddings_540_flat.csv')
def merge_embeddings(df_trailing_stats: pd.DataFrame,
                     df_embeddings: pd.DataFrame,
                     target_feature = 'Jockey') -> pd.DataFrame:
    """
    Merges horse jockey co-ocurrence embedding into 
    race data with trailing stats.
    """

    df_embeddings['Date'] = pd.to_datetime(df_embeddings['Date'])
    df_embeddings.rename(columns={'Target Feature': 'Target_Feature'}, inplace=True)    

    if target_feature == 'Horse':
        for i in range(0, 50):
            df_embeddings.rename(columns={f'{str(i)}': f'H_Emb_{str(i)}'}, inplace=True, errors='ignore')
    else:
        for i in range(0, 50):
            df_embeddings.rename(columns={f'{str(i)}': f'J_Emb_{str(i)}'}, inplace=True, errors='ignore')

    df_merged = pd.merge(df_trailing_stats, df_embeddings,
                        left_on=[target_feature, 'Date'],
                        right_on=['Target_Feature', 'Date'],
                        how='left',
                        suffixes=('', 'z'))

    df_merged.drop(columns=['Target_Feature_y', 
                            'Target_Feature_z',
                            'Date_y',
                            'Date_z'], 
                            inplace=True, errors='ignore')

    # print(f"df_merged: {len(df_merged)}")

    return df_merged

def build_ml_training_data(file_name_horse: str = 'Horse_Horse_Embeddings_2555_all_flat.csv',
                           file_name_jockey: str = 'Jockey_Jockey_Embeddings_2555_all_flat.csv',
                           ) -> None:
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


    df_horse_embeddings = pd.read_csv(f'./data/{file_name_horse}')
    df_jockey_embeddings = pd.read_csv(f'./data/{file_name_jockey}')

    df_merged = merge_embeddings(df_trailing_stats=df,
                                df_embeddings=df_horse_embeddings,
                                target_feature = 'Horse')

    df_merged = merge_embeddings(df_trailing_stats=df_merged,
                                df_embeddings=df_jockey_embeddings,
                                target_feature = 'Jockey')

    cols = df_merged.columns.tolist()
    embedding_cols = [col for col in cols if col[0:6] in ['H_Emb_', 'J_Emb_']]

    return_cols = original_cols + trailing_avg_cols + trailing_avg_cols2 + embedding_cols
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    file_path = os.path.join(script_dir, f"../data/ml_dataset_2019_2025.csv")
    df_merged[return_cols].to_csv(file_path, index=False)

def get_ml_training_data() -> pd.DataFrame:
    return pd.read_csv('./data/ml_dataset_2019_2025.csv')

def join_train_test_files(file_path_base: str,
                          file_path_suffixes: List[str],
                          dt_cut_off_intervals: List[datetime] = None,
                          training_days: int = 120,
                          test_days: int = 30
                          ) -> None:
    """
    Joins all the embedding files into a single file.
    """

    dfs = []
    for suffix, dt_cut_off in zip(file_path_suffixes, dt_cut_off_intervals):
        df = pd.read_csv(r"./data/" + file_path_base + "" + suffix + ".csv")
        print(len(df), "rows in file:", file_path_base + suffix + ".csv")
        df['Effective Date'] = pd.to_datetime(dt_cut_off)
        dfs.append(df)

    df_embeddings_all = pd.concat(dfs, axis=0)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = f"../data/{file_path_base}_{str(training_days)}_train_{str(test_days)}_test_all.csv"
    file_path_joined = os.path.join(script_dir, file_path)
    df_embeddings_all.to_csv(file_path_joined, index=False)



def create_ml_training_data_rank_1() -> pd.DataFrame:
    """
    Returns the joined machine learning training data.
    """

    df_data = pd.read_csv('./data/ml_dataset_2019_2025_390_train_30_test_all.csv')

    conn = sqlite3.connect(':memory:')
    cur = conn.cursor()

    df_data.to_sql('ml_data', conn, index=False)
    sql = """
    WITH ranks AS (
        SELECT 
              RaceNumber , Date , Horse, "Win Odds"
            , RANK() OVER (PARTITION BY RaceNumber, Date ORDER BY "Win Odds" ASC) as odds_rank
            , J_Emb_6, J_Emb_15, J_UMAP_0, J_Emb_26, J_UMAP_4, J_UMAP_5, J_UMAP_3
            , J_UMAP_7, H_Emb_27, H_Emb_33, H_Emb_44, J_Emb_14, J_Emb_19, J_Emb_22
            , J_Emb_28, J_Emb_3, J_Emb_31, J_Emb_33, J_Emb_34

        FROM ml_data
    ),
    rank1 AS (

        SELECT
            m.RaceNumber , m.Date , m.Horse, m.Jockey, m."Win Odds"
            , RANK() OVER (PARTITION BY m.RaceNumber, m.Date ORDER BY m."Win Odds" ASC) as odds_rank
            , m.Placing_TR1, m.Placing_TR2, m.Placing_TR3, m.Placing_TR4, m.Placing_TR5, m.Placing_TR6
            , m.Placing_TR7, m.Placing_TR8, m.Placing_TR9, m.Placing_TR10
            , m.Placing, m."Effective Date", CASE WHEN m.Placing <= 3 THEN 1 ELSE 0 END AS top_three

            , m.J_Emb_6, m.J_Emb_15, m.J_UMAP_0, m.J_Emb_26, m.J_UMAP_4, m.J_UMAP_5, m.J_UMAP_3
            , m.J_UMAP_7, m.H_Emb_27, m.H_Emb_33, m.H_Emb_44, m.J_Emb_14, m.J_Emb_19, m.J_Emb_22
            , m.J_Emb_28, m.J_Emb_3, m.J_Emb_31, m.J_Emb_33, m.J_Emb_34

            , r1.J_Emb_6 AS J_Emb_6_rank_1,	r1.J_Emb_15 AS J_Emb_15_rank_1,	r1.J_UMAP_0 AS J_UMAP_0_rank_1,	r1.J_Emb_26 AS J_Emb_26_rank_1,	r1.J_UMAP_4 AS J_UMAP_4_rank_1,	r1.J_UMAP_5 AS J_UMAP_5_rank_1,	r1.J_UMAP_3 AS J_UMAP_3_rank_1,
            r1.J_UMAP_7 AS J_UMAP_7_rank_1,	r1.H_Emb_27 AS H_Emb_27_rank_1,	r1.H_Emb_33 AS H_Emb_33_rank_1,	r1.H_Emb_44 AS H_Emb_44_rank_1,	r1.J_Emb_14 AS J_Emb_14_rank_1,	r1.J_Emb_19 AS J_Emb_19_rank_1,	r1.J_Emb_22 AS J_Emb_22_rank_1,
            r1.J_Emb_28 AS J_Emb_28_rank_1,	r1.J_Emb_3 AS J_Emb_3_rank_1,	r1.J_Emb_31 AS J_Emb_31_rank_1,	r1.J_Emb_33 AS J_Emb_33_rank_1,	r1.J_Emb_34 AS J_Emb_34_rank_1

            
        FROM ml_data m
            left JOIN ranks r1 ON m.RaceNumber = r1.RaceNumber AND m.Date = r1.Date AND r1.odds_rank = 1
    )

    SELECT * FROM rank1


    """

    #write the tables
    df = pd.read_sql_query(sql, conn)
    file_path_base = "../data/ml_dataset_2019_2025_390_train_30_test_all"
    suffix = "_with_competitor_joins.csv"
    full_file_path = f"{file_path_base}{suffix}"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, full_file_path)
    
    df.to_csv(file_path, index=False)



def create_ml_training_data_rank_2() -> pd.DataFrame:
    """
    Returns the joined machine learning training data.
    """

    df_data_ranks = pd.read_csv('./data/ml_dataset_2019_2025_390_train_30_test_all_with_competitor_joins.csv')

    conn = sqlite3.connect(':memory:')
    cur = conn.cursor()

    df_data_ranks.to_sql('ml_data_ranks', conn, index=False)
    sql = """
    WITH ranks AS (
        SELECT 
              RaceNumber , Date , Horse, "Win Odds", odds_rank
            , m.J_Emb_6, m.J_Emb_15, m.J_UMAP_0, m.J_Emb_26, m.J_UMAP_4, m.J_UMAP_5, m.J_UMAP_3
            , m.J_UMAP_7, m.H_Emb_27, m.H_Emb_33, m.H_Emb_44, m.J_Emb_14, m.J_Emb_19, m.J_Emb_22
            , m.J_Emb_28, m.J_Emb_3, m.J_Emb_31, m.J_Emb_33, m.J_Emb_34

        FROM ml_data_ranks m
    ),
    rank_next AS (

        SELECT
              m.*
            , r1.J_Emb_6 AS J_Emb_6_rank_2,	r1.J_Emb_15 AS J_Emb_15_rank_2,	r1.J_UMAP_0 AS J_UMAP_0_rank_2,	r1.J_Emb_26 AS J_Emb_26_rank_2,	r1.J_UMAP_4 AS J_UMAP_4_rank_2,	r1.J_UMAP_5 AS J_UMAP_5_rank_2,	r1.J_UMAP_3 AS J_UMAP_3_rank_2,
            r1.J_UMAP_7 AS J_UMAP_7_rank_2,	r1.H_Emb_27 AS H_Emb_27_rank_2,	r1.H_Emb_33 AS H_Emb_33_rank_2,	r1.H_Emb_44 AS H_Emb_44_rank_2,	r1.J_Emb_14 AS J_Emb_14_rank_2,	r1.J_Emb_19 AS J_Emb_19_rank_2,	r1.J_Emb_22 AS J_Emb_22_rank_2,
            r1.J_Emb_28 AS J_Emb_28_rank_2,	r1.J_Emb_3 AS J_Emb_3_rank_2,	r1.J_Emb_31 AS J_Emb_31_rank_2,	r1.J_Emb_33 AS J_Emb_33_rank_2,	r1.J_Emb_34 AS J_Emb_34_rank_2			      
        FROM ml_data_ranks m
            left JOIN ranks r1 ON m.RaceNumber = r1.RaceNumber AND m.Date = r1.Date AND r1.odds_rank = 2
    )

    SELECT * FROM rank_next


    """

    #write the tables
    df = pd.read_sql_query(sql, conn)
    file_path_base = "../data/ml_dataset_2019_2025_390_train_30_test_all"
    suffix = "_with_competitor_joins.csv"
    full_file_path = f"{file_path_base}{suffix}"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, full_file_path)
    
    df.to_csv(file_path, index=False)


def create_ml_training_data_rank_3() -> pd.DataFrame:
    """
    Returns the joined machine learning training data.
    """

    df_data_ranks = pd.read_csv('./data/ml_dataset_2019_2025_390_train_30_test_all_with_competitor_joins.csv')

    conn = sqlite3.connect(':memory:')
    cur = conn.cursor()

    df_data_ranks.to_sql('ml_data_ranks', conn, index=False)
    sql = """
    WITH ranks AS (
        SELECT 
              RaceNumber , Date , Horse, "Win Odds", odds_rank
            , m.J_Emb_6, m.J_Emb_15, m.J_UMAP_0, m.J_Emb_26, m.J_UMAP_4, m.J_UMAP_5, m.J_UMAP_3
            , m.J_UMAP_7, m.H_Emb_27, m.H_Emb_33, m.H_Emb_44, m.J_Emb_14, m.J_Emb_19, m.J_Emb_22
            , m.J_Emb_28, m.J_Emb_3, m.J_Emb_31, m.J_Emb_33, m.J_Emb_34

        FROM ml_data_ranks m
    ),
    rank_next AS (

        SELECT
              m.*
            , r1.J_Emb_6 AS J_Emb_6_rank_3,	r1.J_Emb_15 AS J_Emb_15_rank_3,	r1.J_UMAP_0 AS J_UMAP_0_rank_3,	r1.J_Emb_26 AS J_Emb_26_rank_3,	r1.J_UMAP_4 AS J_UMAP_4_rank_3,	r1.J_UMAP_5 AS J_UMAP_5_rank_3,	r1.J_UMAP_3 AS J_UMAP_3_rank_3,
            r1.J_UMAP_7 AS J_UMAP_7_rank_3,	r1.H_Emb_27 AS H_Emb_27_rank_3,	r1.H_Emb_33 AS H_Emb_33_rank_3,	r1.H_Emb_44 AS H_Emb_44_rank_3,	r1.J_Emb_14 AS J_Emb_14_rank_3,	r1.J_Emb_19 AS J_Emb_19_rank_3,	r1.J_Emb_22 AS J_Emb_22_rank_3,
            r1.J_Emb_28 AS J_Emb_28_rank_3,	r1.J_Emb_3 AS J_Emb_3_rank_3,	r1.J_Emb_31 AS J_Emb_31_rank_3,	r1.J_Emb_33 AS J_Emb_33_rank_3,	r1.J_Emb_34 AS J_Emb_34_rank_3			      
        FROM ml_data_ranks m
            left JOIN ranks r1 ON m.RaceNumber = r1.RaceNumber AND m.Date = r1.Date AND r1.odds_rank = 3
    )

    SELECT * FROM rank_next


    """

    #write the tables
    df = pd.read_sql_query(sql, conn)
    file_path_base = "../data/ml_dataset_2019_2025_390_train_30_test_all"
    suffix = "_with_competitor_joins.csv"
    full_file_path = f"{file_path_base}{suffix}"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, full_file_path)
    
    df.to_csv(file_path, index=False)


def create_ml_training_data_rank_4() -> pd.DataFrame:
    """
    Returns the joined machine learning training data.
    """

    df_data_ranks = pd.read_csv('./data/ml_dataset_2019_2025_390_train_30_test_all_with_competitor_joins.csv')

    conn = sqlite3.connect(':memory:')
    cur = conn.cursor()

    df_data_ranks.to_sql('ml_data_ranks', conn, index=False)
    sql = """
    WITH ranks AS (
        SELECT 
              RaceNumber , Date , Horse, "Win Odds", odds_rank
            , m.J_Emb_6, m.J_Emb_15, m.J_UMAP_0, m.J_Emb_26, m.J_UMAP_4, m.J_UMAP_5, m.J_UMAP_3
            , m.J_UMAP_7, m.H_Emb_27, m.H_Emb_33, m.H_Emb_44, m.J_Emb_14, m.J_Emb_19, m.J_Emb_22
            , m.J_Emb_28, m.J_Emb_3, m.J_Emb_31, m.J_Emb_33, m.J_Emb_34

        FROM ml_data_ranks m
    ),
    rank_next AS (

        SELECT
              m.*
            , r1.J_Emb_6 AS J_Emb_6_rank_4,	r1.J_Emb_15 AS J_Emb_15_rank_4,	r1.J_UMAP_0 AS J_UMAP_0_rank_4,	r1.J_Emb_26 AS J_Emb_26_rank_4,	r1.J_UMAP_4 AS J_UMAP_4_rank_4,	r1.J_UMAP_5 AS J_UMAP_5_rank_4,	r1.J_UMAP_3 AS J_UMAP_3_rank_4,
            r1.J_UMAP_7 AS J_UMAP_7_rank_4,	r1.H_Emb_27 AS H_Emb_27_rank_4,	r1.H_Emb_33 AS H_Emb_33_rank_4,	r1.H_Emb_44 AS H_Emb_44_rank_4,	r1.J_Emb_14 AS J_Emb_14_rank_4,	r1.J_Emb_19 AS J_Emb_19_rank_4,	r1.J_Emb_22 AS J_Emb_22_rank_4,
            r1.J_Emb_28 AS J_Emb_28_rank_4,	r1.J_Emb_3 AS J_Emb_3_rank_4,	r1.J_Emb_31 AS J_Emb_31_rank_4,	r1.J_Emb_33 AS J_Emb_33_rank_4,	r1.J_Emb_34 AS J_Emb_34_rank_4			      
        FROM ml_data_ranks m
            left JOIN ranks r1 ON m.RaceNumber = r1.RaceNumber AND m.Date = r1.Date AND r1.odds_rank = 4
    )

    SELECT * FROM rank_next


    """

    #write the tables
    df = pd.read_sql_query(sql, conn)
    file_path_base = "../data/ml_dataset_2019_2025_390_train_30_test_all"
    suffix = "_with_competitor_joins.csv"
    full_file_path = f"{file_path_base}{suffix}"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, full_file_path)
    
    df.to_csv(file_path, index=False)




# ./data/ml_dataset_2019_2025_20250121_120_train_30_test
# ./data/ml_dataset_2019_2025__20250521_120_train_30_test.csv


if __name__ == "__main__":
    # build_ml_training_data(file_name_horse = 'Horse_Horse_Embeddings_2555_all_flat.csv',
    #                        file_name_jockey = 'Jockey_Jockey_Embeddings_2555_all_flat.csv') 



    # create_ml_training_data_rank_1()
    # create_ml_training_data_rank_2()
    # create_ml_training_data_rank_3()
    create_ml_training_data_rank_4()

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




# ,

#     rank3 AS (

#         SELECT
#             m.*,
#             r1.J_Emb_6 AS J_Emb_6_rank_3,	r1.J_Emb_15 AS J_Emb_15_rank_3,	r1.J_UMAP_0 AS J_UMAP_0_rank_3,	r1.J_Emb_26 AS J_Emb_26_rank_3,	r1.J_UMAP_4 AS J_UMAP_4_rank_3,	r1.J_UMAP_5 AS J_UMAP_5_rank_3,	r1.J_UMAP_3 AS J_UMAP_3_rank_3,
#             r1.J_UMAP_7 AS J_UMAP_7_rank_3,	r1.H_Emb_27 AS H_Emb_27_rank_3,	r1.H_Emb_33 AS H_Emb_33_rank_3,	r1.H_Emb_44 AS H_Emb_44_rank_3,	r1.J_Emb_14 AS J_Emb_14_rank_3,	r1.J_Emb_19 AS J_Emb_19_rank_3,	r1.J_Emb_22 AS J_Emb_22_rank_3,
#             r1.J_Emb_28 AS J_Emb_28_rank_3,	r1.J_Emb_3 AS J_Emb_3_rank_3,	r1.J_Emb_31 AS J_Emb_31_rank_3,	r1.J_Emb_33 AS J_Emb_33_rank_3,	r1.J_Emb_34 AS J_Emb_34_rank_3,	r1.J_Emb_41 AS J_Emb_41_rank_3,	r1.J_Emb_42 AS J_Emb_42_rank_3,
#             r1.J_Emb_48 AS J_Emb_48_rank_3,	r1.J_Emb_7 AS J_Emb_7_rank_3,	r1.J_Emb_9 AS J_Emb_9_rank_3,	r1.J_UMAP_2 AS J_UMAP_2_rank_3			            
#         FROM rank2 m
#             JOIN ranks r1 ON m.RaceNumber = r1.RaceNumber AND m.Date = r1.Date AND r1.odds_rank = 3
#     ),
#     rank4 AS (

#         SELECT
#             m.*,
#             r1.J_Emb_6 AS J_Emb_6_rank_4,	r1.J_Emb_15 AS J_Emb_15_rank_4,	r1.J_UMAP_0 AS J_UMAP_0_rank_4,	r1.J_Emb_26 AS J_Emb_26_rank_4,	r1.J_UMAP_4 AS J_UMAP_4_rank_4,	r1.J_UMAP_5 AS J_UMAP_5_rank_4,	r1.J_UMAP_3 AS J_UMAP_3_rank_4,
#             r1.J_UMAP_7 AS J_UMAP_7_rank_4,	r1.H_Emb_27 AS H_Emb_27_rank_4,	r1.H_Emb_33 AS H_Emb_33_rank_4,	r1.H_Emb_44 AS H_Emb_44_rank_4,	r1.J_Emb_14 AS J_Emb_14_rank_4,	r1.J_Emb_19 AS J_Emb_19_rank_4,	r1.J_Emb_22 AS J_Emb_22_rank_4,
#             r1.J_Emb_28 AS J_Emb_28_rank_4,	r1.J_Emb_3 AS J_Emb_3_rank_4,	r1.J_Emb_31 AS J_Emb_31_rank_4,	r1.J_Emb_33 AS J_Emb_33_rank_4,	r1.J_Emb_34 AS J_Emb_34_rank_4,	r1.J_Emb_41 AS J_Emb_41_rank_4,	r1.J_Emb_42 AS J_Emb_42_rank_4,
#             r1.J_Emb_48 AS J_Emb_48_rank_4,	r1.J_Emb_7 AS J_Emb_7_rank_4,	r1.J_Emb_9 AS J_Emb_9_rank_4,	r1.J_UMAP_2 AS J_UMAP_2_rank_4						            
#         FROM rank3 m
#             JOIN ranks r1 ON m.RaceNumber = r1.RaceNumber AND m.Date = r1.Date AND r1.odds_rank = 4
#     )
