"""
General utility functions for the pyphylon package.
"""

import os
import numpy as np
import pandas as pd
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

# NMF normalization #

def _get_normalization_diagonals(W):
    # Generate normalization diagonal matrices
    normalization_vals = [1/np.quantile(W[col], q=0.99) if np.quantile(W[col], q=0.99) > 0 else 0 for col in W.columns]
    recipricol_vals = [1/x if x != 0 else 0 for x in normalization_vals]
    
    D1 = np.diag(normalization_vals)
    D2 = np.diag(recipricol_vals)
    
    return D1, D2
