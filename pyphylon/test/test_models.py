import pytest
import pandas as pd
import numpy as np
from pyphylon.models import *
from sklearn.datasets import load_digits

# TODO change input to functions to datasets such as sklearn's digits dataset (load_digits)
@pytest.fixture
def test_data():
    return pd.DataFrame(load_digits()['data'])

def test_mca(test_data) -> None:
    df = test_data
    mca_results = run_mca(df)
    
    # add any assert statements of interest here
    assert mca_results.n_components == min(test_data.shape)

def test_nmf(test_data) -> None:
    df = test_data
    nmf_w, nmf_h = run_nmf(df, ranks = range(1, int(min(df.shape)*.1)))
    
    # assertion statements of interest
    assert len(nmf_w) == len(list(range(1,int(min(df.shape)*.1))))
    assert len(nmf_h) == len(list(range(1,int(min(df.shape)*.1))))
    assert nmf_w[1].shape == (df.shape[0],1)
    assert nmf_h[1].shape == (1,df.shape[1])

def test_normalization_and_binarization(test_data) -> None:
    df = test_data
    nmf_w, nmf_h = run_nmf(df, ranks = range(1, int(min(df.shape)*.1)))

    L_norm_dict, A_norm_dict = normalize_nmf_outputs(df, nmf_w, nmf_h)
    L_bin_dict, A_bin_dict = binarize_nmf_outputs(L_norm_dict, A_norm_dict)
    
    assert len(L_bin_dict) == len(nmf_w)
    assert max(L_bin_dict[1].max()) == 1
    assert max(A_bin_dict[1].max()) == 1

def test_reconstruction_and_metrics(test_data) -> None:
    df = test_data
    nmf_w, nmf_h = run_nmf(df, ranks = range(1, int(min(df.shape)*.1)))

    L_norm_dict, A_norm_dict = normalize_nmf_outputs(df, nmf_w, nmf_h)
    L_bin_dict, A_bin_dict = binarize_nmf_outputs(L_norm_dict, A_norm_dict)
    # may want to add some test assert statements for correct shape 

    P_reconstructed_dict, P_error_dict, P_confusion_dict = generate_nmf_reconstructions(df, L_bin_dict, A_bin_dict)
    df_metrics = calculate_nmf_reconstruction_metrics(P_reconstructed_dict, P_confusion_dict)

    assert df_metrics.shape[0] == int(min(df.shape)*.1) - 1 # test to make sure there is a metric calculated for every rank
    assert df_metrics.equals(df_metrics.dropna())


def test_run_hdbscan(test_data) -> None:
    df = test_data
    best_model, best_labels, best_model_sil_score, models_df = run_hdbscan(df)

    # add assert statements to test performance of hdbscan