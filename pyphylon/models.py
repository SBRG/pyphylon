"""
Functions for handling dimension-reduction models of pangenome data.
"""

import numpy as np
import pandas as pd
from typing import Iterable
from tqdm.notebook import tqdm, trange
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from prince import MCA
from umap import UMAP
from hdbscan import HDBSCAN

from pyphylon.util import _get_normalization_diagonals

def run_nmf(data, ranks, max_iter=10_000):
    """
    Run NMF multiple times and possibly across multiple ranks.

    Parameters:
    - data: DataFrame containing the dataset to be analyzed.
    - ranks: List of ranks (components) to try.
    - max_iter: Maximum number of iterations to try to reach convergence.

    Returns:
    - W_dict: A dictionary of transformed data at various ranks.
    - H_dict: A dictionary of model components at various ranks.
    """
    W_dict, H_dict = {}, {}

    for rank in tqdm(ranks, desc='Running NMF at varying ranks...'):
        model = NMF(
            n_components=rank,
            init='nndsvd',
            max_iter=max_iter,
            random_state=42
        )
        W = model.fit_transform(data)
        H = model.components_
        W_dict[rank] = W
        H_dict[rank] = H

    return W_dict, H_dict

def normalize_nmf_outputs(data, W_dict, H_dict):
    """
    Normalize NMF outputs (99th percentile of W, column-by-column).

    Parameters:
    - data: Original dataset used for NMF.
    - W_dict: Dictionary containing W matrices.
    - H_dict: Dictionary containing H matrices.

    Returns:
    - L_norm_dict: Normalized L matrices.
    - A_norm_dict: Normalized A matrices.
    """
    L_norm_dict, A_norm_dict = {}, {}
    for rank, W in tqdm(W_dict.items(), desc='Normalizing matrices...'):
        H = H_dict[rank]
        D1, D2 = _get_normalization_diagonals(pd.DataFrame(W))
        L_norm_dict[rank] = pd.DataFrame(np.dot(W, D1), index=data.index)
        A_norm_dict[rank] = pd.DataFrame(np.dot(D2, H), columns=data.columns)
    return L_norm_dict, A_norm_dict

def binarize_nmf_outputs(L_norm_dict, A_norm_dict):
    """
    Binarize NMF outputs using k-means clustering (k=3, top cluster only).

    Parameters:
    - L_norm_dict: Dictionary of normalized L matrices.
    - A_norm_dict: Dictionary of normalized A matrices.

    Returns:
    - L_binarized_dict: Binarized L matrices.
    - A_binarized_dict: Binarized A matrices.
    """
    L_binarized_dict, A_binarized_dict = {}, {}
    for rank in tqdm(L_norm_dict, desc='Binarizing matrices...'):
        L_binarized_dict[rank] = _k_means_binarize_L(L_norm_dict[rank])
        A_binarized_dict[rank] = _k_means_binarize_A(A_norm_dict[rank])
    return L_binarized_dict, A_binarized_dict

def generate_nmf_reconstructions(data, L_binarized_dict, A_binarized_dict):
    '''
    Calculate model reconstr, error, & confusion matrix for each L_bin & A_bin
    '''
    P_reconstructed_dict = {}
    P_error_dict = {}
    P_confusion_dict = {}

    for rank in tqdm(
        L_binarized_dict,
        desc='Evaluating model reconstructions...'
    ):
        reconstr, err, confusion = _calculate_nmf_reconstruction(
            data,
            L_binarized_dict[rank],
            A_binarized_dict[rank]
        )

        P_reconstructed_dict[rank] = reconstr
        P_error_dict[rank] = err
        P_confusion_dict[rank] = confusion
    
    return P_reconstructed_dict, P_error_dict, P_confusion_dict

def calculate_nmf_reconstruction_metrics(
        P_reconstructed_dict,
        P_confusion_dict
    ):
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

    Parameters:
    - data: DataFrame containing the dataset to be analyzed.

    Returns:
    - MCA fitted model.
    """
    mca = MCA(
        n_components=min(data.shape),
        n_iter=1,
        copy=True,
        check_input=True,
        engine='sklearn',
        random_state=42
    )
    return mca.fit(data)

def run_polytope_vertex_group_extraction(
        data,
        low_memory=False,
        core_dist_n_jobs=8
    ):
    """
    Run DensMAP + HDBSCAN on the (binary) dataset (describing a polytope).

    Parameters:
    - data: DataFrame containing the dataset to be analyzed.
    - low_memory: Boolean, passed onto UMAP to optimize memory usage.
    - core_dist_n_jobs: Integer, num of jobs for core dist calcs in UMAP.

    Returns:
    - Cluster labels from HDBSCAN.
    """
    densmap = UMAP(
        n_components=3,
        n_neighbors=0.01 * min(data.shape),
        metric='cosine',
        min_dist=0.0,
        random_state=42,
        densmap=True,
        low_memory=low_memory
    )
    # TODO: add in sensitivity analysis here
    embedding = densmap.fit_transform(data)
    clusterer = HDBSCAN(min_cluster_size=15, metric='euclidean')
    return clusterer.fit_predict(embedding)

class NmfModel(object):
    '''
    Class representation of NMF models and their reconstructions w/metrics
    '''

    def __init__(
            self,
            data: pd.DataFrame,
            ranks: Iterable,
            max_iter: int = 10_000
        ):
        '''
        Initialize the NmfModel object w/ required data matrix and rank list.

        Parameters:
        - data: DataFrame on which NMF will be run
        - ranks: Iterable of ranks on which to perform NMF
        - max_iter: Integer, Max num of iters for convergence, default 10_000
        '''

        self._data = data
        self._ranks = ranks
        self._max_iter = max_iter

        # Initialize other properties to None
        self._W_dict = None
        self._H_dict = None
        self._L_norm_dict = None
        self._A_norm_dict = None
        self._L_binarized_dict = None
        self._A_binarized_dict = None
        self._P_reconstructed_dict = None
        self._P_error_dict = None
        self._P_confusion_dict = None
        self._df_metrics = None

    
    @property
    def data(self):
        '''Get input data for NMF models'''
        return self._data
    
    @property
    def ranks(self):
        '''Get ranks on which NMF will be performed'''
        return self._ranks
    
    # If None, compute the following properties when called
    
    @property
    def W_dict(self):
        '''Get a dictionary of raw W matrices across chosen ranks'''
        if not self._W_dict:
            W_dict, H_dict = run_nmf(
                self._data,
                self._ranks,
                max_iter=self._max_iter
            )
            self._W_dict = W_dict
            self._H_dict = H_dict
        
        return self._W_dict
    
    @W_dict.setter
    def W_dict(self, new_dict):
        self._W_dict = new_dict

    @property
    def H_dict(self):
        '''Get a dictionary of raw H matrices across chosen ranks'''
        if not self._H_dict:
            W_dict, H_dict = run_nmf(
                self._data,
                self._ranks,
                max_iter=self._max_iter
            )
            self._W_dict = W_dict
            self._H_dict = H_dict
        
        return self._H_dict
    
    @H_dict.setter
    def H_dict(self, new_dict):
        self._H_dict = new_dict
    
    @property
    def L_norm_dict(self):
        '''Get a dictionary of L matrices across chosen ranks'''
        if not self._L_norm_dict:
            L_norm_dict, A_norm_dict = normalize_nmf_outputs(
                self.data,
                self.W_dict,
                self.H_dict
            )
            self._L_norm_dict = L_norm_dict
            self._A_norm_dict = A_norm_dict
        
        return self._L_norm_dict
    
    @L_norm_dict.setter
    def L_norm_dict(self, new_dict):
        self._L_norm_dict = new_dict

    @property
    def A_norm_dict(self):
        '''Get a dictionary of A matrices across chosen ranks'''
        if not self._A_norm_dict:
            L_norm_dict, A_norm_dict = normalize_nmf_outputs(
                self.data,
                self.W_dict,
                self.H_dict
            )
            self._L_norm_dict = L_norm_dict
            self._A_norm_dict = A_norm_dict
        
        return self._A_norm_dict
    
    @A_norm_dict.setter
    def A_norm_dict(self, new_dict):
        self._A_norm_dict = new_dict

    @property
    def L_binarized_dict(self):
        '''Get a dictionary of binarized L matrices across chosen ranks'''
        if not self._L_binarized_dict:
            L_binarized_dict, A_binarized_dict = binarize_nmf_outputs(
                self.L_norm_dict,
                self.A_norm_dict
            )
            self._L_binarized_dict = L_binarized_dict
            self._A_binarized_dict = A_binarized_dict
        
        return self._L_binarized_dict
    
    @L_binarized_dict.setter
    def L_binarized_dict(self, new_dict):
        self._L_binarized_dict = new_dict

    @property
    def A_binarized_dict(self):
        '''Get a dictionary of binarized A matrices across chosen ranks'''
        if not self._A_binarized_dict:
            L_binarized_dict, A_binarized_dict = binarize_nmf_outputs(
                self.L_norm_dict,
                self.A_norm_dict
            )
            self._L_binarized_dict = L_binarized_dict
            self._A_binarized_dict = A_binarized_dict
        
        return self._A_binarized_dict
    
    @A_binarized_dict.setter
    def A_binarized_dict(self, new_dict):
        self._A_binarized_dict = new_dict
    
    @property
    def P_reconstructed_dict(self):
        '''
        Get a dictionary of the reconstructed data from post-processed NMF.
        '''
        if not self._P_reconstructed_dict:
            reconstr, error, confusion = generate_nmf_reconstructions(
                self.data,
                self.L_binarized_dict,
                self.A_binarized_dict
            )
            self._P_reconstructed_dict = reconstr
            self._P_error_dict = error
            self._P_confusion_dict = confusion
            
        return self._P_reconstructed_dict
    
    @P_reconstructed_dict.setter
    def P_reconstructed_dict(self, new_dict):
         self._P_reconstructed_dict = new_dict
    
    @property
    def P_error_dict(self):
        '''
        Get a dictionary of errors between orig and reconstr data matrices.
        '''
        if not self._P_error_dict:
            reconstr, error, confusion = generate_nmf_reconstructions(
                self.data,
                self.L_binarized_dict,
                self.A_binarized_dict
            )
            self._P_reconstructed_dict = reconstr
            self._P_error_dict = error
            self._P_confusion_dict = confusion
                
        return self._P_error_dict
    
    @P_error_dict.setter
    def P_error_dict(self, new_dict):
         self._P_error_dict = new_dict
    
    @property
    def P_confusion_dict(self):
        '''
        Get a dictionary of the confusion matrix.
        '''
        if not self._P_confusion_dict:
            reconstr, error, confusion = generate_nmf_reconstructions(
                self.data,
                self.L_binarized_dict,
                self.A_binarized_dict
            )
            self._P_reconstructed_dict = reconstr
            self._P_error_dict = error
            self._P_confusion_dict = confusion
        
        return self._P_confusion_dict
    
    @P_confusion_dict.setter
    def P_confusion_dict(self, new_dict):
         self._P_confusion_dict = new_dict
    
    @property
    def df_metrics(self):
        '''
        Return a table of metrics for NMF model reconstructions across ranks.
        '''
        if not self._df_metrics:
            df_metrics = calculate_nmf_reconstruction_metrics(
                self.P_reconstructed_dict,
                self.P_confusion_dict
            )
            self._df_metrics = df_metrics
        
        return self._df_metrics
    
    @df_metrics.setter
    def df_metrics(self, new_dict):
         self.df_metrics = new_dict

# Helper functions
def _k_means_binarize_L(L_norm):
    '''
    Use k-means clustering (k=3) to binarize L_norm matrix.
    '''
    
    # Initialize an empty array to hold the binarized matrix
    L_binarized = np.zeros_like(L_norm.values)
    
    # Loop through each column
    for col_idx in trange(
        L_norm.values.shape[1],
        leave=False,
        desc='binarizing column by column...'
    ):
        column_data = L_norm.values[:, col_idx]
    
        # Reshape the column data to fit the KMeans input shape
        column_data_reshaped = column_data.reshape(-1, 1)
    
        # Apply 3-means clustering (gen better P-R tradeoff than 2-means)
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
    L_binarized = pd.DataFrame(
        L_binarized,
        index=L_norm.index,
        columns=L_norm.columns
    )
    return L_binarized


def _k_means_binarize_A(A_norm):
    '''
    Use k-means clustering (k=3) to binarize A_norm matrix.
    '''
    # Initialize an empty array to hold the binarized matrix
    A_binarized = np.zeros_like(A_norm.values)
    
    # Loop through each row
    for row_idx in trange(
        A_norm.values.shape[0],
        leave=False,
        desc='binarizing row by row...'
    ):
        row_data = A_norm.values[row_idx, :]
    
        # Reshape the row data to fit the KMeans input shape
        row_data_reshaped = row_data.reshape(-1, 1)
    
        # Apply 3-means clustering (gen better P-R tradeoff than 2-means)
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
    A_binarized = pd.DataFrame(
        A_binarized,
        index=A_norm.index,
        columns=A_norm.columns
    )
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
    
    # Binarize the orig and reconstr matrices for confusion matrix calculation
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
    P_confusion = confusion_matrix(
        data_flat,
        P_reconstructed_flat,
        labels=[1, 0]
    )
    
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
    P_plus_R = Precision + Recall
    FPR = FP / (FP + TN) if FP + TN != 0 else 0
    FNR = FN / (TP + FN) if TP + FN != 0 else 0
    Specificity = TN / (TN + FP) if TN + FP != 0 else 0
    Prevalence = (TP + FN) / (TP + TN + FP + FN)
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    F1_score = 2 * (Precision * Recall) / (P_plus_R) if P_plus_R != 0 else 0
    BM = Recall + Specificity - 1

    # Adjusted MCC calculation to avoid overflow
    numerator = TP * TN - FP * FN
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    MCC = numerator / denominator if denominator != 0 else 0

    Jaccard_index = TP / (TP + FP + FN) if TP + FP + FN != 0 else 0
    one_minus_Sp = 1 - Specificity
    if BM != 0:
        PT = (np.sqrt(Recall * one_minus_Sp) + Specificity - 1) / (BM)
    else:
        PT = 0
    
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
        'Prevalence Threshold': PT,
        'MCC': MCC,
        'Jaccard Index': Jaccard_index
    }
