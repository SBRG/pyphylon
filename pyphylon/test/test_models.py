import pytest
import pandas as pd
import numpy as np
from pyphylon.models import *


def test_mca() -> None:
    df = pd.DataFrame(np.random.randint(0,2,size=(20,5)))
    mca_results = run_mca
    # add any assert statements of interest here

def test_nmf() -> None:
    df = pd.DataFrame(np.random.randint(0,2,size=(20,5)))
    nmf_w, nmf_h = run_nmf(df, ranks = range(1, min(df.shape) + 1))
    # may want to add some test assert statements for correct shape

def test_normalization_and_binarization() -> None:
    df = pd.DataFrame(np.random.randint(0,2,size=(20,5)))
    nmf_w, nmf_h = run_nmf(df, ranks = range(1, min(df.shape) + 1))

    L_norm_dict, A_norm_dict = normalize_nmf_outputs(df, nmf_w, nmf_h)
    L_bin_dict, A_bin_dict = binarize_nmf_outputs(L_norm_dict, A_norm_dict)
    # may want to add some test assert statements for correct shape 


def test_reconstruction_and_metrics() -> None:
    df = pd.DataFrame(np.random.randint(0,2,size=(20,5)))
    nmf_w, nmf_h = run_nmf(df, ranks = range(1, min(df.shape) + 1))

    L_norm_dict, A_norm_dict = normalize_nmf_outputs(df, nmf_w, nmf_h)
    L_bin_dict, A_bin_dict = binarize_nmf_outputs(L_norm_dict, A_norm_dict)
    # may want to add some test assert statements for correct shape 

    P_reconstructed_dict, P_error_dict, P_confusion_dict = generate_nmf_reconstructions(df, L_bin_dict, A_bin_dict)
    df_metrics = calculate_nmf_reconstruction_metrics(P_reconstructed_dict, P_confusion_dict)

    assert df_metrics.shape[0] == min(df.shape)

def test_run_hdbscan() -> None:
    df = pd.DataFrame(np.random.randint(0,2,size=(100,50)))
    best_model, best_labels, best_model_sil_score, models_df = run_hdbscan(df)