"""
General utility functions for the pyphylon package.
"""

def _validate_identical_shapes(mat1, mat2):
    if mat1.shape != mat2.shape:
        raise ValueError(
            f"Dimension mismatch. {mat1.shape} and {mat2.shape} must have the same dimensions."
        )

def _validate_decomposition_shapes(input_mat, output1, output2):
    if input_mat.shape[0] != output1.shape[0]:
        raise ValueError(
            f"Dimension mismatch. {input_mat.shape} and {output1.shape} must have the same number of rows."
        )
    if input_mat.shape[1] != output2.shape[1]:
        raise ValueError(
            f"Dimension mismatch. {input_mat.shape} and {output2.shape} must have the same number of columns."
        )
    if output1.shape[1] != output2.shape[0]:
        raise ValueError(
            f"Dimension mismatch. Number of columns in {output1.shape} must match number of rows in {output2.shape}."
        )
