"""
General utility functions for the pyphylon package.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm.notebook import trange

# Data shape validation #

def _validate_identical_shapes(mat1, mat2, name1, name2):
    if mat1.shape != mat2.shape:
        raise ValueError(
            f"Dimension mismatch. {name1} {mat1.shape} and {name2} {mat2.shape} must have the same dimensions."
        )

def _validate_decomposition_shapes(input_mat, output1, output2, input_name, output1_name, output2_name):
    if input_mat.shape[0] != output1.shape[0]:
        raise ValueError(
            f"Dimension mismatch. {input_name} {input_mat.shape} and {output1_name} {output1.shape} must have the same number of rows."
        )
    if input_mat.shape[1] != output2.shape[1]:
        raise ValueError(
            f"Dimension mismatch. {input_name} {input_mat.shape} and {output2_name} {output2.shape} must have the same number of columns."
        )
    if output1.shape[1] != output2.shape[0]:
        raise ValueError(
            f"Dimension mismatch. Number of columns in {output1_name} {output1.shape} must match number of rows in {output2_name} {output2.shape}."
        )

# NMF normalization & binarization #

def _get_normalization_diagonals(W):
    '''
    Generate normalization matrices (to normalize W & H into L & A)
    '''
    normalization_vals = [1/np.quantile(W[col], q=0.99) for col in W.columns]
    recipricol_vals = [1/x for x in normalization_vals]
    
    D1 = np.diag(normalization_vals)
    D2 = np.diag(recipricol_vals)
    
    return D1, D2

def k_means_binarize_L(L_norm):
    '''
    Use k-means clustering (k=3) to binarize L_norm and A_norm matrices
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


def k_means_binarize_A(A_norm):
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

