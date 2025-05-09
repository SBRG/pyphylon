"""
General utility functions for the pyphylon package.
"""

import os
import numpy as np
import pandas as pd

from scipy.optimize import nnls
from joblib import Parallel, delayed
from tqdm.notebook import trange

# Files and folders #

def load_config(config_file):
    # Load configuration file
    import yaml
    with open(config_file, 'r') as stream:
        CONFIG = yaml.safe_load(stream)
    return CONFIG

def remove_empty_files(directory):
    # Initialize list of empty files
    empty_files = []

    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            if size == 0:
                empty_files.append(file)
                os.remove(file_path)
        else:
            continue

    return empty_files

# Data validation #

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

def _check_and_convert_binary_sparse(df: pd.DataFrame):
    # Validate matrix is binary
    if not df.astype("int8").isin([0, 1]).all().all():
        raise ValueError("The DataFrame is not binary. It contains values other than 0 / 1 or False / True.")

    # Ensure matrix is stored as a SparseDtype of int8
    df = _convert_sparse(df, "int8")

    return df

def _convert_sparse(df: pd.DataFrame, dtype='int8'):
    # Ensure matrix is stored as a SparseDtype of int8 or float
    cond1 = all(pd.api.types.is_sparse(df[col]) for col in df.columns)
    cond2 = df.dtypes.unique().tolist() != [pd.SparseDtype("int8", 0)]

    if not cond1 or cond2:
        if dtype == 'int8':
            df = df.astype(pd.SparseDtype("int8", 0))
        else:
            df = df.astype(pd.SparseDtype("float"))
    
    return df

def infer_affinities(L: np.ndarray, P_new: np.ndarray, n_jobs: int = -1) -> np.ndarray:
    """
    Infer affinities for new genomes by solving a non-negative least squares problem in parallel.
    
    Given a binary gene presence/absence matrix P_new (with genes as rows and genomes as columns)
    and a precomputed basis matrix L (with genes as rows and phylons as columns), this function
    computes A_new (with phylons as rows and genomes as columns) such that:
    
        P_new ≈ L @ A_new
        
    For each genome (column in P_new), the following NNLS problem is solved:
    
        a_new = argmin_{a >= 0} || L @ a - p ||²
        
    Parallelization is used to speed up computations across multiple CPU cores.
    
    Parameters
    ----------
    L : np.ndarray
        A 2D numpy array of shape (n_genes, n_phylons) representing the basis (or "phylon" signatures)
        derived from non-negative matrix factorization.
    P_new : np.ndarray
        A 2D numpy array of shape (n_genes, n_genomes) representing the new binary gene 
        presence/absence data.
    n_jobs : int, optional
        The number of jobs to run in parallel. Defaults to -1, which uses all available cores.
        
    Returns
    -------
    A_new : np.ndarray
        A 2D numpy array of shape (n_phylons, n_genomes) representing the inferred affinities (or 
        activity levels) for the new genomes.
    
    Notes
    -----
    This function solves an independent non-negative least squares (NNLS) problem for each genome
    to ensure that the resulting affinities are non-negative, preserving the NMF constraints.
    The computation is parallelized across genomes to accelerate processing.
    
    Examples
    --------
    >>> import numpy as np
    >>> L = np.array([[0.5, 0.3], [0.2, 0.7]])
    >>> P_new = np.array([[1, 0], [0, 1]])
    >>> A_new = infer_affinities(L, P_new, n_jobs=2)
    >>> print(A_new)
    [[1. 0.]
     [0. 1.]]
    """
    # Get dimensions and validate that the gene counts match.
    n_genes, n_phylons = L.shape
    if P_new.shape[0] != n_genes:
        raise ValueError("The number of rows (genes) in P_new must match the number in L.")
    
    n_genomes = P_new.shape[1]
    
    def solve_nnls(i: int) -> np.ndarray:
        """
        Solve the NNLS problem for the i-th genome.
        
        Parameters
        ----------
        i : int
            Index of the genome (column in P_new) for which to solve the NNLS.
            
        Returns
        -------
        a : np.ndarray
            The inferred affinity vector for the i-th genome.
        """
        p = P_new[:, i]
        a, _ = nnls(L, p)
        return a
    
    # Compute NNLS for each genome in parallel.
    results = Parallel(n_jobs=n_jobs)(
        delayed(solve_nnls)(i) for i in trange(n_genomes)
    )
    
    # Stack the results (each is a vector of length n_phylons) into a matrix.
    A_new = np.column_stack(results)
    
    return A_new

def _get_normalization_diagonals(W):
    # Generate normalization diagonal matrices
    normalization_vals = [1/np.quantile(W[col], q=0.99) for col in W.columns]
    recipricol_vals = [1/x for x in normalization_vals]
    
    D1 = np.diag(normalization_vals)
    D2 = np.diag(recipricol_vals)
    
    return D1, D2



