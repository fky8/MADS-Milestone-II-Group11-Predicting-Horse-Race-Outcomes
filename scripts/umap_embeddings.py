import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import trustworthiness
from typing import List
from collections import defaultdict

# https://umap-learn.readthedocs.io/en/latest/basic_usage.html


def umap_sensitivity_graph(scaled_data: np.ndarray,
                           neighbors_ls: List[float], 
                           min_dist_ls: List[float], 
                           connectivity_ls: List[float]) -> None:
    """
    Perform UMAP sensitivity analysis by varying n_neighbors, min_dist, and local_connectivity.
    """

    dict_neighbors = defaultdict(float)
    dict_dist = defaultdict(float)
    dict_connectivity = defaultdict(float)

    static_neighbors = 5
    static_min_dist = 1.0
    static_connectivity = 0.25

    for n in neighbors_ls:
        reducer = umap.UMAP(init='spectral', learning_rate=0.1, 
                            local_connectivity=static_connectivity,
                            metric='cosine', min_dist=static_min_dist, 
                            n_components=10, n_neighbors=n, n_jobs=-1)

        embedding = reducer.fit_transform(scaled_data)
        trust_score = trustworthiness(scaled_data, embedding, n_neighbors=n)
        dict_neighbors[n] = trust_score

    n_neighbors_values = []
    trustworthiness_scores = []
    for n, score in dict_neighbors.items():
        n_neighbors_values.append(np.log(n))
        trustworthiness_scores.append(score)

    plt.figure(figsize=(10, 6))
    plt.plot(n_neighbors_values, trustworthiness_scores, marker='o', linestyle='-')
    plt.title('Hyperparameter Sensitivity: n_neighbors vs. Trustworthiness')
    plt.xlabel('log(n_neighbors)')
    plt.ylabel('Trustworthiness Score')
    plt.grid(True)
    plt.show()

        # min_dist_values.append(np.log(m))
        # local_connectivity_values.append(np.log(c))

    for m in min_dist_ls:
        reducer = umap.UMAP(init='spectral', learning_rate=0.1, 
                            local_connectivity=static_connectivity,
                            metric='cosine', min_dist=m, 
                            n_components=10, n_neighbors=static_neighbors, 
                            n_jobs=-1)

        embedding = reducer.fit_transform(scaled_data)
        trust_score = trustworthiness(scaled_data, embedding, n_neighbors=static_neighbors)
        dict_dist[m] = trust_score

    min_dist_values = []
    # local_connectivity_values = []
    dist_scores = []
    for m, score in dict_dist.items():
        min_dist_values.append(np.log(m))
        dist_scores.append(score)


    plt.figure(figsize=(10, 6))
    plt.plot(min_dist_values, dist_scores, marker='o', linestyle='-')
    plt.title('Hyperparameter Sensitivity: min_dist vs. Trustworthiness')
    plt.xlabel('log(min_dist)')
    plt.ylabel('Trustworthiness Score')
    plt.grid(True)
    plt.show()


    for c in connectivity_ls:
        reducer = umap.UMAP(init='spectral', learning_rate=0.1, 
                            local_connectivity=c,
                            metric='cosine', min_dist=static_min_dist, 
                            n_components=10, n_neighbors=static_neighbors, 
                            n_jobs=-1)

        embedding = reducer.fit_transform(scaled_data)
        trust_score = trustworthiness(scaled_data, embedding, n_neighbors=static_neighbors)
        dict_connectivity[c] = trust_score

    min_dist_values = []
    # local_connectivity_values = []
    dist_scores = []
    for c, score in dict_connectivity.items():
        min_dist_values.append(np.log(c))
        dist_scores.append(score)


    plt.figure(figsize=(10, 6))
    plt.plot(min_dist_values, dist_scores, marker='o', linestyle='-')
    plt.title('Hyperparameter Sensitivity: local_connectivity vs. Trustworthiness')
    plt.xlabel('log(local_connectivity)')
    plt.ylabel('Trustworthiness Score')
    plt.grid(True)
    plt.show()

def umap_sensitivity_matrix(scaled_data: np.ndarray,
                            hyper_parameter_1: str = 'n_neighbors', 
                            hyper_parameter_2: str = 'min_dist',
                            hyper_parameter_ls_1: List[float] = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
                            hyper_parameter_ls_2: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],                            
                            default_n_neighbors: int = 15,
                            default_min_dist: float = 0.1,
                            default_local_connectivity: float = 0.25
                            ) -> None:
    """
    Creates a UMAP sensitivity matrix by comparing 2 out of 3 hyperparameters
    chosen hyperparameters. Examples include: n_neighbors, min_dist, and local_connectivity.
    """

    accepted_hyper_parameters = ['n_neighbors', 'min_dist', 'local_connectivity']
    
    list_hyper_params = [hyper_parameter_1, hyper_parameter_2]

    dict_index = {
        'n_neighbors': None,
        'min_dist': None,
        'local_connectivity': None
    }

    dict_hyper_defautl = {
        'n_neighbors': default_n_neighbors,
        'min_dist': default_min_dist,
        'local_connectivity': default_local_connectivity
    }
    
    dict_hyper_param = {
        'n_neighbors': False,
        'min_dist': False,
        'local_connectivity': False
    }

    dict_hyper_param_values = {
        'n_neighbors': [],
        'min_dist': [],
        'local_connectivity': []
    }



    if hyper_parameter_1 not in accepted_hyper_parameters or \
       hyper_parameter_2 not in accepted_hyper_parameters:
        raise ValueError(f"Accepted hyperparameters are: {accepted_hyper_parameters}")
    else:
        dict_hyper_param[hyper_parameter_1] = True
        dict_hyper_param[hyper_parameter_2] = True
        dict_hyper_param_values[hyper_parameter_1] = hyper_parameter_ls_1
        dict_hyper_param_values[hyper_parameter_2] = hyper_parameter_ls_2
        dict_index[hyper_parameter_1] = 0
        dict_index[hyper_parameter_2] = 1


    dict_matrix = defaultdict(float)

    nn = 'n_neighbors'
    md = 'min_dist'
    lc = 'local_connectivity'
    
    for i in range(0, len(hyper_parameter_ls_1)):
        for j in range(0, len(hyper_parameter_ls_2)):
        
            nn_i = i if dict_index[nn] == 0 else j
            md_i = i if dict_index[md] == 0 else j
            cc_i = i if dict_index[lc] == 0 else j
        
            local_connectivity_value = dict_hyper_param_values[lc][cc_i] if dict_hyper_param[lc] else dict_hyper_defautl[lc]

            n_neighbors_value = dict_hyper_param_values[nn][nn_i] if dict_hyper_param[nn] else dict_hyper_defautl[nn]
            min_dist_value = dict_hyper_param_values[md][md_i] if dict_hyper_param[md] else dict_hyper_defautl[md]

            reducer = umap.UMAP(init='spectral', learning_rate=0.1, 
                                local_connectivity=local_connectivity_value,
                                metric='cosine', min_dist=min_dist_value,
                                n_components=10, n_neighbors=n_neighbors_value, n_jobs=-1)

            embedding = reducer.fit_transform(scaled_data)
            trust_score = trustworthiness(scaled_data, embedding, n_neighbors=n_neighbors_value)
            dict_matrix[(n_neighbors_value, min_dist_value, local_connectivity_value)] = trust_score


    for (n, m, c), score in dict_matrix.items():
        print(str(n) + "," + str(m) + "," + str(c) + "," + str(score))

def build_umap_embeddings(scaled_data,
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



if __name__ == "__main__":

    # df_data = pd.read_csv('./data/Jockey_Jockey_Embeddings_2555.csv')
    # cols = [
    # '0',	'1',	'2',	'3',	'4',	'5',	'6',	'7',	
    # '8',	'9',	'10',	'11',	'12',	'13',	'14',	'15',	'16',	'17',	
    # '18',	'19',	'20',	'21',	'22',	'23',	'24',	'25',	'26',	'27',	
    # '28',	'29',	'30',	'31',	'32',	'33',	'34',	'35',	'36',	'37',	
    # '38',	'39',	'40',	'41',	'42',	'43',	'44',	'45',	'46',	'47',	
    # '48',	'49',
    # ]

    # scaler = StandardScaler()
    # scaled_data = scaler.fit_transform(df_data[cols])
    # run_analysis = False

    # if  not run_analysis:
    #     # Final chosen model was a compromise between trustworthiness and the ability of the model 
    #     # to produce effective clusters for downstream ml tasks.
    #     build_umap_embeddings(scaled_data,
    #                         n_components = 10,
    #                         n_neighbors = 15,
    #                         min_dist = 0.50,
    #                         local_connectivity = 0.75)
    # else:

        # # Hyperparameter sensitivity search parameters
        # neighbors_ls = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        # min_dist_ls = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # connectivity_ls = [0.5, 1.0, 1.5, 2.0, 2.5, 3.00, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

        # # Graph model sensitivity to n_neighbors, min_dist, and local_connectivity hyperparameters
        # umap_sensitivity_graph(scaled_data=scaled_data, neighbors_ls=neighbors_ls, 
        #                     min_dist_ls=min_dist_ls, 
        #                     connectivity_ls=connectivity_ls)

        # # Produce a comparison matrix of hyperparameters n_neighbors, min_dist
        # umap_sensitivity_matrix(scaled_data=scaled_data,
        #                         hyper_parameter_1 = 'n_neighbors', 
        #                         hyper_parameter_2 = 'min_dist',
        #                         hyper_parameter_ls_1 = neighbors_ls, 
        #                         hyper_parameter_ls_2 = min_dist_ls) 

        # # highest trustworthiness scores of 0.80 were values n_neighbors = 5 and min_dist = 1.0
        # # lowest scores were n_neighbors = 60 and min_dist = 0.1
        # # Decided to examine parameters between 15 and 20 for n_neighbors and 0.3 to 0.9 for min_dist

        # # Analze connectivity vs. n_neighbors hyperparameters
        # umap_sensitivity_matrix(scaled_data=scaled_data,
        #                         hyper_parameter_1 = 'n_neighbors', 
        #                         hyper_parameter_2 = 'local_connectivity',
        #                         hyper_parameter_ls_1 = neighbors_ls, 
        #                         hyper_parameter_ls_2 = connectivity_ls,
        #                         default_n_neighbors = 15,
        #                         default_min_dist = 0.30,
        #                         default_local_connectivity = 0.25                        
        #                         ) 

        # # best candidates appear below 1.0, indicating tighter clusters preserve trustworthiness

        # # Analyze connectivity below 1.0
        # connectivity_ls = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
        # umap_sensitivity_matrix(scaled_data=scaled_data,
        #                         hyper_parameter_1 = 'n_neighbors', 
        #                         hyper_parameter_2 = 'local_connectivity',
        #                         hyper_parameter_ls_1 = neighbors_ls, 
        #                         hyper_parameter_ls_2 = connectivity_ls,
        #                         default_n_neighbors = 15,
        #                         default_min_dist = 0.30,
        #                         default_local_connectivity = 0.25                        
        #                         ) 
        # # Final chosen model was a compromise between trustworthiness and the ability of the model 
        # # to produce effective clusters for downstream ml tasks.

    df_data = pd.read_csv('./data/Horse_Horse_Embeddings_2555.csv')
    cols = [
    '0',	'1',	'2',	'3',	'4',	'5',	'6',	'7',	
    '8',	'9',	'10',	'11',	'12',	'13',	'14',	'15',	'16',	'17',	
    '18',	'19',	'20',	'21',	'22',	'23',	'24',	'25',	'26',	'27',	
    '28',	'29',	'30',	'31',	'32',	'33',	'34',	'35',	'36',	'37',	
    '38',	'39',	'40',	'41',	'42',	'43',	'44',	'45',	'46',	'47',	
    '48',	'49',
    ]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_data[cols])
    run_analysis = False
    
    neighbors_ls = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    min_dist_ls = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    connectivity_ls = [0.5, 1.0, 1.5, 2.0, 2.5, 3.00, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

    # Produce a comparison matrix of hyperparameters n_neighbors, min_dist
    # umap_sensitivity_matrix(scaled_data=scaled_data,
    #                         hyper_parameter_1 = 'n_neighbors', 
    #                         hyper_parameter_2 = 'min_dist',
    #                         hyper_parameter_ls_1 = neighbors_ls, 
    #                         hyper_parameter_ls_2 = min_dist_ls) 

    # highest trustworthiness scores of 0.80 were values n_neighbors = 5 and min_dist = 1.0
    # lowest scores were n_neighbors = 60 and min_dist = 0.1
    # Decided to examine parameters between 15 and 20 for n_neighbors and 0.3 to 0.9 for min_dist

    # # Analze connectivity vs. n_neighbors hyperparameters
    # umap_sensitivity_matrix(scaled_data=scaled_data,
    #                         hyper_parameter_1 = 'n_neighbors', 
    #                         hyper_parameter_2 = 'local_connectivity',
    #                         hyper_parameter_ls_1 = neighbors_ls,
    #                         hyper_parameter_ls_2 = connectivity_ls,
    #                         default_n_neighbors = 15,
    #                         default_min_dist = 0.10,
    #                         default_local_connectivity = 0.25) 

    # # best candidates appear below 1.0, indicating tighter clusters preserve trustworthiness

    # # Analyze connectivity below 1.0
    connectivity_ls = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    umap_sensitivity_matrix(scaled_data=scaled_data,
                            hyper_parameter_1 = 'n_neighbors', 
                            hyper_parameter_2 = 'local_connectivity',
                            hyper_parameter_ls_1 = neighbors_ls, 
                            hyper_parameter_ls_2 = connectivity_ls,
                            default_n_neighbors = 15,
                            default_min_dist = 0.30,
                            default_local_connectivity = 0.25                       
                            )


####################
### Scratch Code ###
####################

    # reducer = umap.UMAP(
    #                 a=None, angular_rp_forest=False, b=None,
    #                 force_approximation_algorithm=False,
    #                 init='spectral',
    #                 learning_rate=0.1,
    #                 local_connectivity=0.25,
    #                 low_memory=False,
    #                 metric='cosine',
    #                 #  metric_kwds=None,
    #                 min_dist=1.0,
    #                 n_components=10,
    #                 #  n_epochs=None,
    #                 n_neighbors=5, 
    #                 #  negative_sample_rate=5, 
    #                 #  output_metric='cosine',
    #                 output_metric_kwds=None, 
    #                 # random_state=777, 
    #                 #  repulsion_strength=1.0,
    #                 #  set_op_mix_ratio=1.0, 
    #                 #  spread=1.0, 
    #                 #  target_metric='categorical',
    #                 #  target_metric_kwds=None, 
    #                 #  target_n_neighbors=-1, 
    #                 #  target_weight=0.5,
    #                 #  transform_queue_size=4.0, 
    #                 #  transform_seed=42, 
    #                 unique=False, 
    #                 verbose=False,
    #             n_jobs=-1
    #      )



    # embedding = reducer.fit_transform(scaled_data)
    # target = df_data['Target Feature'].astype('category').cat.codes
    # plt.scatter(embedding[:, 0], embedding[:, 1], c=target, cmap='tab10', s=20, alpha=0.8)
    # labels = df_data['Target Feature'].tolist()
    # for i, txt in enumerate(labels):
    #     plt.annotate(txt, 
    #                  (embedding[i, 0], embedding[i, 1]), 
    #                  textcoords="offset points", 
    #                  xytext=(0,10), 
    #                  ha='center')

    # trust_score = trustworthiness(scaled_data, embedding, n_neighbors=5)
    # print(f'Trustworthiness score: {trust_score:.4f}')

    # plt.colorbar(label='Target Feature')
    # plt.title('UMAP Jockey Embeddings')
    # plt.xlabel('Dim 1')
    # plt.ylabel('Dim 2')
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.show()
