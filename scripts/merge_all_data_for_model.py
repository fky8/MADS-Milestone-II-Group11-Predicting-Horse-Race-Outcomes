from datetime import datetime, timedelta
import sys
import pandas as pd
from typing import List
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.dirname(current_dir)
sys.path.append(scripts_dir)

from scripts.query_repository import get_race_data_with_trailing_stats, get_race_dataset, join_train_test_files

if __name__ == "__main__":


    # '2025-05-21'
    # '2024-07-25'
    dt_cut_off_str = '2024-07-25'
    training_interval_days = [390]
    test_interval_days = [30]
    dt_cut_off_intervals = []
    suffixes = []

    for i in range(0, 12):
        dt_cut_off_intervals.append(datetime.strptime(dt_cut_off_str, '%Y-%m-%d') - timedelta(days=i*30))

    df_horse_embeddings = pd.read_csv('./data/Horse_Horse_Embeddings_2555_all_flat.csv')
    df_jockey_embeddings = pd.read_csv('./data/Jockey_Jockey_Embeddings_2555_all_flat.csv')

    df_horse_embeddings['Date'] = pd.to_datetime(df_horse_embeddings['Date'])
    df_horse_embeddings.rename(columns={'Target Feature': 'Target_Feature'}, inplace=True)    
    df_jockey_embeddings['Date'] = pd.to_datetime(df_jockey_embeddings['Date'])
    df_jockey_embeddings.rename(columns={'Target Feature': 'Target_Feature'}, inplace=True)    

    df_race = get_race_dataset(start_date='2019-01-01', end_date='2025-12-31')
    original_cols = df_race.columns.tolist()

    df = get_race_data_with_trailing_stats(groupby='Horse')


    for training_days in training_interval_days:    
        for test_days in test_interval_days:
            for dt_cut_off in dt_cut_off_intervals:

                # dt_cut_off_str = '2025-05-21'
                dt_cut_off_str = dt_cut_off.strftime('%Y-%m-%d')
                file_suffix = '_' + dt_cut_off_str.replace('-', '') + '_' + str(training_days) + '_train_' + str(test_days) + '_test'
                suffixes.append(file_suffix)
                date_cut_off_test = datetime.strptime(dt_cut_off_str, '%Y-%m-%d') - timedelta(days=test_days)
                date_cut_off_embs = date_cut_off_test - timedelta(days=training_days)



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

                df_filtered = df[df['Date'] >= date_cut_off_embs]
                df_horse_embeddings_next = df_horse_embeddings[df_horse_embeddings['Date'] == date_cut_off_embs]
                df_jockey_embeddings_next = df_jockey_embeddings[df_jockey_embeddings['Date'] == date_cut_off_embs]

                for i in range(0, 50):
                    df_horse_embeddings_next.rename(columns={f'{str(i)}': f'H_Emb_{str(i)}'}, inplace=True, errors='ignore')
  
                df_merged = pd.merge(df_filtered, df_horse_embeddings_next,
                                    left_on=['Horse'],
                                    right_on=['Target_Feature'],
                                    how='left',
                                    suffixes=('', 'z'))

                df_merged.drop(columns=['Target_Feature_y', 
                                        'Target_Feature_z',
                                        'Date_y',
                                        'Date_z'], 
                                        inplace=True, errors='ignore')

                for i in range(0, 50):
                    df_jockey_embeddings_next.rename(columns={f'{str(i)}': f'J_Emb_{str(i)}'}, inplace=True, errors='ignore')

                df_merged = pd.merge(df_merged, df_jockey_embeddings_next,
                                    left_on=['Jockey'],
                                    right_on=['Target_Feature'],
                                    how='left',
                                    suffixes=('', 'z'))

                df_merged.drop(columns=['Target_Feature_y', 
                                        'Target_Feature_z',
                                        'Date_y',
                                        'Date_z'], 
                                        inplace=True, errors='ignore')


                cols = df_merged.columns.tolist()
                embedding_cols = [col for col in cols if col[0:6] in ['H_Emb_', 'J_Emb_', 'H_UMAP', 'J_UMAP']]

                return_cols = original_cols + trailing_avg_cols + trailing_avg_cols2 + embedding_cols
                script_dir = os.path.dirname(os.path.abspath(__file__)) 
                file_path = os.path.join(script_dir, f"../data/ml_dataset_2019_2025{file_suffix}.csv")
                df_merged[return_cols].to_csv(file_path, index=False)

    # Combine files into a single file=

    join_train_test_files(file_path_base='ml_dataset_2019_2025',
                        file_path_suffixes=suffixes,
                        dt_cut_off_intervals=dt_cut_off_intervals,
                        training_days=training_days,
                        test_days=test_days)
    
    print('suffixes', suffixes)
    print('dt_cut_off_intervals', str(dt_cut_off_intervals))
    print('training_days', str(training_days))
    print('test_days', str(test_days))

