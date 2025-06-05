

import os
import pandas as pd

import time
import re


def create_trailing_average_win_stats(groupby='Horse', metric_grouping='Score range'):
    """
    Aggregate the jockey data to itself to calculate cumulative win stats 
    for every race date.
    Returns a DataFrame with aggregated jockey data for win stats.
    """

    df = pd.read_csv('./data/Race_comments_gear_horse_competitors_2019_2025.csv')
    print("df row count start: ", len(df))
    #  filter out horses without a placing due to withdrawal or disqualification
    df['Placing'] = pd.to_numeric(df['Placing'], errors='coerce')
    df_filtered = df[df['Placing'].notna()]    
    print("df row count filter non placing horses: ", len(df_filtered))


    # Get only the fields we are interested in creating the cumulative stats
    # The going referes to condition of the surface of the track
    df_filtered['DistanceMeterAsStr'] = df_filtered['DistanceMeter'].astype(str)
    df_filtered['Start Time'] =  pd.Timestamp.today().normalize()
    df_filtered['Finish Time'] = pd.to_datetime(df_filtered['Finish Time'])
    df_filtered['Finish Time In Seconds'] = (df_filtered['Finish Time'] - df_filtered['Start Time']).dt.total_seconds() / 60
    df_filtered['RunningPosition1'] = df_filtered['RunningPosition1'].astype(float)
    df_filtered['RunningPosition2'] = df_filtered['RunningPosition2'].astype(float)
    df_filtered['RunningPosition3'] = df_filtered['RunningPosition3'].astype(float)
    df_filtered['RunningPosition4'] = df_filtered['RunningPosition4'].astype(float)
    df_filtered['RunningPosition5'] = df_filtered['RunningPosition5'].astype(float)
    df_filtered['RunningPosition6'] = df_filtered['RunningPosition6'].astype(float)    
    df_filtered['Date'] = pd.to_datetime(df_filtered['Date'])
    df_filtered['Count'] = 1
    
    gropyby_columns = ['Horse', 'Jockey', 'Dr.', 'Date', 'DistanceMeterAsStr', 
                       'Score range', 'Going'
                       # 'Age', Need to find a way to calculate age from date of birth given horse data
                       ]
    
    stat_columns = ['Finish Time In Seconds', 'Placing' ]        
    df_filtered = df_filtered[gropyby_columns + stat_columns]
    
    encode_columns = [metric_grouping]

    df_encoded = pd.get_dummies(df_filtered, columns=encode_columns, dtype=int, )
    df_encoded = pd.concat([df_filtered[metric_grouping], df_encoded], axis=1)
    df_encoded = df_encoded.sort_values(by=[groupby, metric_grouping, 'Date'], ascending=[True, True, True])

    columns = gropyby_columns + stat_columns
    set1 = set(columns)
    set2 = set(df_encoded.columns.tolist())
    unique_encoded_columns = list(set2 - set1)

    # trail_average_columns = ['Finish Time In Seconds']

    unique_encoded_value_columns = []
    for col in stat_columns:
        for col2 in unique_encoded_columns:
        
            col_name = col2 + ' ' + col  + ' Value'
            unique_encoded_value_columns.append(col_name)
            # make value none so it is not included in average is there gaps in the attribute
            df_encoded[col_name] = df_encoded.apply(lambda x: None if x[col2] == 0 or x[col2] is None else x[col] * x[col2], axis=1) 
            df_encoded[col_name] = df_encoded.groupby([groupby, metric_grouping])[col_name].bfill()
            df_encoded[col_name] = df_encoded.groupby([groupby, metric_grouping])[col_name].ffill()


    window_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    new_cols = []
    for col in unique_encoded_value_columns:
        for w in window_sizes:
            new_col = df_encoded.groupby([groupby, metric_grouping], as_index=False)[col].shift(1).transform(lambda x: x.rolling(window=w).mean())
            new_col_name = col + ' TR' + str(w)
            new_col = new_col.rename(new_col_name)
            new_cols.append(new_col)

    df_encoded.drop(columns=unique_encoded_value_columns, inplace=True)
    df_encoded.drop(columns=unique_encoded_columns, inplace=True)


    df_new_cols = pd.concat(new_cols, axis=1)
    df_encoded_new = pd.concat([df_encoded.copy(), df_new_cols], axis=1)

    # fill in the missing trailing averages

    window_sizes_reversed = window_sizes[0:len(window_sizes) - 1][::-1]
    for w2 in window_sizes[::-1]:
        for w in window_sizes_reversed:
            for col in unique_encoded_value_columns:
                col_name_1 = col + ' TR' + str(w2)
                col_name_2 = col + ' TR' + str(w)
                df_encoded_new.loc[df_encoded_new[col_name_1].isna(), col_name_1] = df_encoded_new[col_name_2]

    df_encoded_new.dropna(axis=1, how='all', inplace=True)

    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    file_path = os.path.join(script_dir, f"data/" + groupby + "_trailing_stats_" + metric_grouping +".csv")
    df_encoded_new.to_csv(file_path, index=False)



# groupby_columns = ['Horse', 'Jockey']
groupby_columns = ['Jockey']
metrics = ['Dr.', 'DistanceMeterAsStr', 'Score range', 'Going']

for col in groupby_columns:
    for metric in metrics:
        create_trailing_average_win_stats(col, metric)