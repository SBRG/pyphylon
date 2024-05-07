'''
Handling reduced-dimension models of P
'''

import multiprocessing as mp
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import NMF
from sklearn.metrics import confusion_matrix
from prince import MCA
from umap import UMAP
from tqdm.notebook import tqdm, trange

from pyphylon.util import _get_normalization_diagonals

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

def normalize_nmf_outputs(data, W_dict, H_dict):
    '''
    Normalize NMF outputs (99th perctentile = 1, column-by-column)
    '''
    L_norm_dict = {}
    A_norm_dict = {}

    for rank, matrix in tqdm(W_dict.items(), desc='Normalizing matrices...'):
        D1, D2 = _get_normalization_diagonals(pd.DataFrame(matrix))
        
        L_norm_dict[rank] = pd.DataFrame(np.dot(W_dict[rank], D1), index=data.index)
        A_norm_dict[rank] = pd.DataFrame(np.dot(D2, H_dict[rank]), columns=data.columns)
    
    return L_norm_dict, A_norm_dict

def binarize_nmf_outputs(L_norm_dict, A_norm_dict):
    '''
    Binarize NMF outputs (k-means clustering, k=3, top cluster only)
    '''
    L_binarized_dict = {}
    A_binarized_dict = {}

    for rank in tqdm(L_norm_dict , desc='Binarizing matrices...'):
        L_binarized_dict[rank] = _k_means_binarize_L(L_norm_dict[rank])
        A_binarized_dict[rank] = _k_means_binarize_A(A_norm_dict[rank])
    
    return L_binarized_dict, A_binarized_dict

def generate_nmf_reconstructions(data, L_binarized_dict, A_binarized_dict):
    '''
    Calculate the model reconstruction, error, and confusion matrix for each L & A decomposition
    '''
    P_reconstructed_dict = {}
    P_error_dict = {}
    P_confusion_dict = {}

    for rank in tqdm(L_binarized_dict, desc='Evaluating model reconstructions...'):
        P_reconstructed_dict[rank], P_error_dict[rank], P_confusion_dict[rank] = _calculate_nmf_reconstruction(
            data,
            L_binarized_dict[rank],
            A_binarized_dict[rank]
        )
    
    return P_reconstructed_dict, P_error_dict, P_confusion_dict

def calculate_nmf_reconstruction_metrics(P_reconstructed_dict, P_confusion_dict):
    '''
    Calculate all reconstruction metrics from the generated confusion matrix
    '''
    df_metrics = pd.DataFrame()

    for rank in tqdm(P_reconstructed_dict, desc='Tabulating metrics...'):
        df_metrics[rank] = _calculate_metrics(P_confusion_dict[rank])
    
    df_metrics = df_metrics.T
    df_metrics.index.name = 'rank'

    return df_metrics

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

# Helper functions
def _k_means_binarize_L(L_norm):
    '''
    Use k-means clustering (k=3) to binarize L_norm matrix
    '''
    
    # Initialize an empty array to hold the binarized matrix
    L_binarized = np.zeros_like(L_norm.values)
    
    # Loop through each column
    for col_idx in trange(L_norm.values.shape[1], leave=False, desc='binarizing column by column...'):
        column_data = L_norm.values[:, col_idx]
    
        # Reshape the column data to fit the KMeans input shape
        column_data_reshaped = column_data.reshape(-1, 1)
    
        # Apply 3-means clustering (generally better precision-recall tradeoff than 2-means)
        kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto')
        kmeans.fit(column_data_reshaped)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
    
        # Find the cluster with the highest mean
        highest_mean_cluster = np.argmax(centers)
    
        # Binarize the column based on the cluster with the highest mean
        binarized_column = (labels == highest_mean_cluster).astype(int)
    
        # Update the binarized matrix
        L_binarized[:, col_idx] = binarized_column
    
    # Typecast to DataFrame
    L_binarized = pd.DataFrame(L_binarized, index=L_norm.index, columns=L_norm.columns)
    return L_binarized


def _k_means_binarize_A(A_norm):
    '''
    Use k-means clustering (k=3) to binarize A_norm matrix
    '''
    # Initialize an empty array to hold the binarized matrix
    A_binarized = np.zeros_like(A_norm.values)
    
    # Loop through each row
    for row_idx in trange(A_norm.values.shape[0], leave=False, desc='binarizing row by row...'):
        row_data = A_norm.values[row_idx, :]
    
        # Reshape the row data to fit the KMeans input shape
        row_data_reshaped = row_data.reshape(-1, 1)
    
        # Apply 3-means clustering (generally better precision-recall tradeoff than 2-means)
        kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto')
        kmeans.fit(row_data_reshaped)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
    
        # Find the cluster with the highest mean
        highest_mean_cluster = np.argmax(centers)
    
        # Binarize the row based on the cluster with the highest mean
        binarized_row = (labels == highest_mean_cluster).astype(int)
    
        # Update the binarized matrix
        A_binarized[row_idx, :] = binarized_row
    
    # Typecast to DataFrame
    A_binarized = pd.DataFrame(A_binarized, index=A_norm.index, columns=A_norm.columns)
    return A_binarized

def _calculate_nmf_reconstruction(data, L_binarized, A_binarized):
    
    # Multiply the binarized matrices to get the reconstructed matrix
    P_reconstructed = pd.DataFrame(
        np.dot(L_binarized, A_binarized),
        index=data.index,
        columns=data.columns
    )
    
    # Calculate the error matrix
    P_error = data - P_reconstructed
    
    # Binarize the original and reconstructed matrices for confusion matrix calculation
    data_binary = (data.values > 0).astype('int8')
    P_reconstructed_binary = (P_reconstructed.values > 0).astype('int8')
    
    # Flatten the matrices to use them in the confusion matrix calculation
    data_flat = data_binary.flatten()
    P_reconstructed_flat = P_reconstructed_binary.flatten()
    
    # Generate the confusion matrix
    # Definitions:
    # True Positive (TP): both actual and predicted are true
    # False Positive (FP): actual is false, but predicted is true
    # True Negative (TN): both actual and predicted are false
    # False Negative (FN): actual is true, but predicted is false
    P_confusion = confusion_matrix(data_flat, P_reconstructed_flat, labels=[1, 0])
    
    return P_reconstructed, P_error, P_confusion

def _calculate_metrics(P_confusion):
    
    # Unpack confusion matrix elements
    TP = P_confusion[0, 0]
    FN = P_confusion[0, 1]
    FP = P_confusion[1, 0]
    TN = P_confusion[1, 1]
    
    # Use float for calculations to prevent integer overflow
    TP, FN, FP, TN = map(float, [TP, FN, FP, TN])
    
    # Calculations
    Precision = TP / (TP + FP) if TP + FP != 0 else 0
    Recall = TP / (TP + FN) if TP + FN != 0 else 0
    FPR = FP / (FP + TN) if FP + TN != 0 else 0
    FNR = FN / (TP + FN) if TP + FN != 0 else 0
    Specificity = TN / (TN + FP) if TN + FP != 0 else 0
    Prevalence = (TP + FN) / (TP + TN + FP + FN)
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    F1_score = 2 * (Precision * Recall) / (Precision + Recall) if Precision + Recall != 0 else 0
    BM = Recall + Specificity - 1

    # Adjusted MCC calculation to avoid overflow
    numerator = TP * TN - FP * FN
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    MCC = numerator / denominator if denominator != 0 else 0

    Jaccard_index = TP / (TP + FP + FN) if TP + FP + FN != 0 else 0
    Prevalence_Threshold = (np.sqrt(Recall * (1 - Specificity)) + Specificity - 1) / (Recall + Specificity - 1) if Recall + Specificity - 1 != 0 else 0
    
    return {
        'Precision': Precision,
        'Recall': Recall,
        'FPR': FPR,
        'FNR': FNR,
        'Specificity': Specificity,
        'Prevalence': Prevalence,
        'Accuracy': Accuracy,
        'F1 Score': F1_score,
        'BM': BM,
        'Prevalence Threshold': Prevalence_Threshold,
        'MCC': MCC,
        'Jaccard Index': Jaccard_index
    }
