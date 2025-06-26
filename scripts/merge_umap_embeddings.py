
import os
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import trustworthiness
from typing import List

# the lower local_connectivity, the more it emphasizes local structure
# min_dist .50 or higher helps to preserve the global structure of the data
# n_neighbors 15 helps identify local structure in the data

def build_jockey_umap_embeddings(scaled_data,
                          n_components: int = 10,
                          n_neighbors: int = 15,
                          min_dist: float = 0.50,
                          local_connectivity: float = 0.75) -> tuple(): 

    reducer = umap.UMAP(init='spectral', learning_rate=0.1, 
                        local_connectivity=local_connectivity,
                        metric='cosine', min_dist=min_dist, 
                        n_components=n_components, n_neighbors=n_neighbors, 
                        n_jobs=-1)

    embedding = reducer.fit_transform(scaled_data)
    trust_score = trustworthiness(scaled_data, embedding, n_neighbors=n_neighbors)

    return embedding, trust_score


def build_horse_umap_embeddings(scaled_data,
                          n_components: int = 10,
                          n_neighbors: int = 15,
                          min_dist: float = 0.10,
                          local_connectivity: float = 0.90) -> tuple():

    reducer = umap.UMAP(init='spectral', learning_rate=0.1, 
                        local_connectivity=local_connectivity,
                        metric='cosine', min_dist=min_dist, 
                        n_components=n_components, n_neighbors=n_neighbors, 
                        n_jobs=-1)

    embedding = reducer.fit_transform(scaled_data)
    trust_score = trustworthiness(scaled_data, embedding, n_neighbors=n_neighbors)

    return embedding, trust_score



if __name__ == "__main__":


    build_horse_umaps = True
    build_jockey_umaps = False

    if build_horse_umaps:

        df = pd.read_csv('./data/Horse_Horse_Embeddings_2555_all.csv')
        cols = [str(i) for i in range(10)]
        col_names = ["H_UMAP_" + str(col) for col in cols]
        list_dates = df['Date End'].unique().tolist()[::-1]
        dfs = pd.DataFrame()

        for date in list_dates:      
            
            df_next = df[df['Date End'] == pd.to_datetime(date).strftime('%Y-%m-%d')]
            df_labels = df_next['Target Feature'].copy()
            df_next.drop(columns=['Target Feature', 'Date End', 'Date Begin'], inplace=True, errors='ignore')

            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df_next)


            emb, score = build_horse_umap_embeddings(scaled_data)

            df_emb = pd.DataFrame(emb, columns=col_names)
            df_emb['Date'] = date
            df_emb['Target Feature'] = df_labels.values
            df_emb['Trust Score'] = score

            dfs = pd.concat([dfs, df_emb], axis=0)

        df_merged = pd.merge(df, dfs,
                    left_on=['Target Feature', 'Date End'],
                    right_on=['Target Feature', 'Date'],
                    how='left',
                    suffixes=('', '_y'))

        df_merged.drop(columns=['Target Feature_y', 'Date End_y', 'Date_y', 'Date'], inplace=True, errors='ignore')

        script_dir = os.path.dirname(os.path.abspath(__file__)) 
        file_path = os.path.join(script_dir, f"../data/Horse_Horse_Embeddings_2555_all.csv")
        df_merged.to_csv(file_path, index=False)    

    elif build_jockey_umaps:

        df = pd.read_csv('./data/Jockey_Jockey_Embeddings_2555_all.csv')
        cols = [str(i) for i in range(10)]
        col_names = ["J_UMAP_" + str(col) for col in cols]
        list_dates = df['Date End'].unique().tolist()[::-1]
        dfs = pd.DataFrame()

        for date in list_dates:      
            
            df_next = df[df['Date End'] == pd.to_datetime(date).strftime('%Y-%m-%d')]
            df_labels = df_next['Target Feature'].copy()
            df_next.drop(columns=['Target Feature', 'Date End', 'Date Begin'], inplace=True, errors='ignore')

            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df_next)


            emb, score = build_jockey_umap_embeddings(scaled_data,
                                    n_components = 10,
                                    n_neighbors = 15,
                                    min_dist = 0.50,
                                    local_connectivity = 0.75)

            df_emb = pd.DataFrame(emb, columns=col_names)
            df_emb['Date'] = date
            df_emb['Target Feature'] = df_labels.values
            df_emb['Trust Score'] = score

            dfs = pd.concat([dfs, df_emb], axis=0)

        df_merged = pd.merge(df, dfs,
                    left_on=['Target Feature', 'Date End'],
                    right_on=['Target Feature', 'Date'],
                    how='left',
                    suffixes=('', '_y'))

        df_merged.drop(columns=['Target Feature_y', 'Date End_y', 'Date_y', 'Date'], inplace=True, errors='ignore')

        script_dir = os.path.dirname(os.path.abspath(__file__)) 
        file_path = os.path.join(script_dir, f"../data/Jockey_Jockey_Embeddings_2555_all.csv")
        df_merged.to_csv(file_path, index=False)