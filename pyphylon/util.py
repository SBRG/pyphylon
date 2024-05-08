"""
General utility functions for the pyphylon package.
"""

import numpy as np
import pandas as pd
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

# NMF normalization #

def _get_normalization_diagonals(W):
    '''
    Generate normalization matrices (to normalize W & H into L & A)
    '''
    normalization_vals = [1/np.quantile(W[col], q=0.99) for col in W.columns]
    recipricol_vals = [1/x for x in normalization_vals]
    
    D1 = np.diag(normalization_vals)
    D2 = np.diag(recipricol_vals)
    
    return D1, D2