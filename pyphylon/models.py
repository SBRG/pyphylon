'''
Handling reduced-dimension models of P
'''

import multiprocessing as mp
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import NMF
from prince import MCA
from umap import UMAP
from tqdm.notebook import tqdm

def run_nmf(data, ranks, max_iter=10_000):
    """
    Run NMF multiple times and possibly across multiple ranks.

    :param data: DataFrame containing the dataset to be analyzed.
    :param ranks: List of ranks (components) to try.
    :param max_iter: Max number of iterations to try to reach convergence.
    :return W_dict: A dictionary of transformed data at various ranks.
    :return H_dict: A dictionary of model components at various ranks.
    """
    W_dict = {}
    H_dict = {}

    # Perform NMF for each rank in ranks
    for rank in tqdm(ranks, desc='Running NMF at varying ranks...'):
        
        model = NMF(n_components=rank,
                    init='nndsvd',      # Run NMF with NNDSVD initialization (for sparsity)
                    max_iter=max_iter,
                    random_state=42
                    )
        
        W = model.fit_transform(data)
        H = model.components_
        reconstruction = np.dot(W, H)
        error = np.linalg.norm(data - reconstruction, 'fro')  # Calculate the Frobenius norm of the difference

        # Store the best W and H matrices in the dictionaries with the rank as the key
        W_dict[rank] = W
        H_dict[rank] = H

        return W_dict, H_dict

def run_mca(data):
    """
    Run Multiple Correspondence Analysis (MCA) on the dataset.

    :param data: DataFrame containing the dataset to be analyzed.
    :return: MCA fitted model.
    """
    mca = MCA(n_components=2, random_state=42)
    mca.fit(data)
    return mca

def run_densmap_hdbscan(data):
    """
    Run DensMAP followed by HDBSCAN on the dataset.

    :param data: DataFrame containing the dataset to be analyzed.
    :return: Cluster labels from HDBSCAN.
    """
    densmap = UMAP(n_components=2, n_neighbors=30, min_dist=0.0, metric='euclidean', random_state=42, densmap=True)
    embedding = densmap.fit_transform(data)
    clusterer = HDBSCAN(min_cluster_size=15, metric='euclidean')
    labels = clusterer.fit_predict(embedding)
    return labels
