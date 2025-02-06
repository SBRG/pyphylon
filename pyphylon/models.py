"""
Functions for handling dimension-reduction models of pangenome data.
"""

import logging
from pyexpat import model
import numpy as np
import pandas as pd
from typing import Iterable, Union, List, Tuple, Dict, Any, Optional
from tqdm.notebook import tqdm, trange
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, silhouette_score
from prince import MCA
from umap import UMAP
from hdbscan import HDBSCAN

from pyphylon.util import _get_normalization_diagonals

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

################################
#           Functions          #
################################

# Multiple Corresspondence Analysis (MCA)
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

# Non-negative Matrix Factorization (NMF)
def run_nmf(data: Union[np.ndarray, pd.DataFrame], ranks: List[int], max_iter: int = 10_000):
    """
    Run NMF on the input data across multiple ranks.
    NMF decomposes a non-negative matrix D into two non-negative matrices W and H:
    D ≈ W * H

    Where:
    - D is the input data matrix (n_samples, n_features)
    - W is the basis matrix (n_samples, n_components)
    - H is the coefficient matrix (n_components, n_features)

    The optimization problem solved is:
    min_{W,H} 0.5 * ||D - WH||_Fro^2
    subject to W,H >= 0

    Non-negative Double Singlular Value Decomposition (NNDSVD) is used
    for the intialization of the optimization problem. This is done to
    ensure the basis matrix (and correspondingly the coefficient matrix)
    is as sparse as possible.

    Parameters:
    - data: DataFrame containing the dataset to be analyzed.
    - ranks: List of ranks (components) to try.
    - max_iter: Maximum number of iterations to try to reach convergence.

    Returns:
    - W_dict: A dictionary of transformed data at various ranks.
    - H_dict: A dictionary of model components at various ranks.

    Notes:
    ------
    This function uses the 'nndsvd' initialization, which is based on two SVD processes,
    one approximating the data matrix, the other approximating positive sections of 
    the resulting partial SVD factors.

    References:
    -----------
    Boutsidis, C., & Gallopoulos, E. (2008). SVD based initialization: A head start 
    for nonnegative matrix factorization. Pattern Recognition, 41(4), 1350-1362.
    """
    # Data validation
    if data.ndim != 2:
        raise ValueError("data must be a 2-dimensional array")
    if not all(r > 0 for r in ranks):
        raise ValueError("ranks must be a list of positive integers")
    if max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")

    # Initialize outputs
    W_dict, H_dict = {}, {}

    # Run NMF at varying ranks
    logger.info(f"Starting NMF process for {len(ranks)} ranks")
    for rank in tqdm(ranks, desc='Running NMF at varying ranks...'):
        model = NMF(
            n_components=rank,
            init='nndsvd',
            max_iter=max_iter,
            random_state=42
        )

        logger.debug(f"Fitting NMF model for rank {rank}")
        W = model.fit_transform(data)
        H = model.components_
        W_dict[rank] = W
        H_dict[rank] = H

    logger.info("NMF process completed for given ranks")
    return W_dict, H_dict

def normalize_nmf_outputs(data: pd.DataFrame, 
                          W_dict: Dict[int, np.ndarray], 
                          H_dict: Dict[int, np.ndarray]) -> Tuple[Dict[int, pd.DataFrame], Dict[int, pd.DataFrame]]:
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
        try:
            H = H_dict[rank]
            D1, D2 = _get_normalization_diagonals(pd.DataFrame(W))
            L_norm_dict[rank] = pd.DataFrame(np.dot(W, D1), index=data.index)
            A_norm_dict[rank] = pd.DataFrame(np.dot(D2, H), columns=data.columns)
        except KeyError:
            logging.warning(f"Rank {rank} not found in H_dict. Skipping...") # TODO: update to long-form logging
        except ValueError as e:
            logging.error(f"Error normalizing matrices for rank {rank}: {str(e)}") # TODO: update to long-form logging
            raise
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
    """
    Calculate model reconstr, error, & confusion matrix for each L_bin & A_bin
    """
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

# Calculate model reconstruction metrics
def calculate_nmf_reconstruction_metrics(
        P_reconstructed_dict,
        P_confusion_dict
    ):
    """
    Calculate all reconstruction metrics from the generated confusion matrix
    """
    df_metrics = pd.DataFrame()

    for rank in tqdm(P_reconstructed_dict, desc='Tabulating metrics...'):
        df_metrics[rank] = _calculate_metrics(P_confusion_dict[rank], P_reconstructed_dict[rank], rank)
    
    df_metrics = df_metrics.T
    df_metrics.index.name = 'rank'

    return df_metrics

# Polytope Vertex Group Extraction (PVGE)
def run_densmap(
        data: pd.DataFrame,
        low_memory: bool = False,
        n_neighbors: int = None
    ):
    """
    Run DensMAP for density-preserving, nonlinear dimension reduction.

    This reduction can be run on `data` as well as its transpose `data.T`.

    Parameters:
    - data (pd.DataFrame): Data to be embedded for dimension-reduction.
    - low_memory (bool): Passed onto UMAP to optimize memory usage.
    - n_neighbors (int): Passed onto UMAP to determine local/global dim. red.

    Returns:
    - densmap (UMAP): A DensMAP model object fitted to data.
    - embedding (pd.DataFrame): A DataFrame of the reduced embedding.
    """
    if n_neighbors:
        n_neighbors = _check_n_neighbors(data, n_neighbors)
    else:
        n_neighbors = 0.01 * min(data.shape)
    
    if len(embedding) == 0:
        raise ValueError("Empty embedding array provided")

    if len(embedding) == 1:
        logging.warning("Only one point provided. Returning single-cluster result.")
        return HDBSCAN(), np.array([0]), 1.0, pd.DataFrame()
    
    densmap = UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        metric='cosine',
        min_dist=0.0,
        random_state=42,
        densmap=True,
        low_memory=low_memory
    )

    embedding = densmap.fit_transform(data)
    return densmap, embedding

def run_hdbscan(
        embedding: np.ndarray, 
        max_range: Optional[int] = None, 
        core_dist_n_jobs: int = 8
        ) -> Tuple[Any, np.ndarray, float, pd.DataFrame]:
    """
    Run HDBSCAN across various cluster sizes and sample sizes.

    Returns a multi-indexed DataFrame of models and their relevant
    metrics.

    Parameters:
    - embedding (pd.DataFrame): Dimensionally-reduced dataset to cluster
    - max_range (int): max range for hyperparameter tuning
    - core_dist_n_jobs (int): Num of parallel jobs to run for core dist calcs

    Returns:
    - best_model (HDBSCAN): the best fitting model
    - best_labels (np.array): the clustering label predictions of best_model
    - df_models_with_metrics (pd.DataFrame): DataFrame of models with metrics
    """
    # Define ranges for HDBSCAN parameters to tune
    max_size = 0.05 * max(embedding.shape)
    if max_range < 100:
        max_size = 100
    else:
        max_size = max_range
    
    min_cluster_sizes = np.linspace(start=5, stop=max_size, num=5).astype(int)
    min_samples_range = np.linspace(start=5, stop=max_size, num=5).astype(int)
    
    # Initialize criteria for comparing HDBSCAN models
    best_relative_validity = -1
    best_model = None
    best_labels = None

    # Create a MultiIndex DataFrame to store results
    index = pd.MultiIndex.from_product(
        iterables=[min_cluster_sizes, min_samples_range],
        names=['min_cluster_size', 'min_samples']
    )
    models_df = pd.DataFrame(
        index=index,
        columns=['model', 'labels', 'relative_validity', 'silhouette_score']
    )

    # Iterate over combinations of min_cluster_size and min_samples
    for min_cluster_size in tqdm(
        min_cluster_sizes,
        desc='Tuning over min. cluster sizes'
    ):
        for min_samples in tqdm(
            min_samples_range,
            desc='Tuning over min. sample sizes',
            leave=False
        ):
            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',
                core_dist_n_jobs=core_dist_n_jobs
            )
            labels = clusterer.fit_predict(embedding)

            # Evaluate clustering if > 1 cluster and noise (-1) < 50%
            if len(set(labels)) > 1 and np.count_nonzero(labels != -1) / len(labels) > 0.5:
                score = silhouette_score(embedding, labels)
                if clusterer.relative_validity_ > best_relative_validity:
                    best_model = clusterer
                    best_labels = labels
                    best_model_sil_score = score
                    best_relative_validity = clusterer.relative_validity_
            
            else:
                score = -1
            
            models_df.loc[(min_cluster_size, min_samples), 'model'] = clusterer
            models_df.loc[(min_cluster_size, min_samples), 'labels'] = labels
            models_df.loc[(min_cluster_size, min_samples), 'relative_validity'] = clusterer.relative_validity_
            models_df.loc[(min_cluster_size, min_samples), 'silhouette_score'] = score
    
    return best_model, best_labels, best_model_sil_score, models_df

################################
#           Classes            #
################################

# Container for NMF models for easy loading into NmfData
class NmfModel(object):
    """
    Class representation of NMF models and their reconstructions w/metrics
    """

    def __init__(
            self,
            data: pd.DataFrame,
            ranks: Iterable,
            max_iter: int = 10_000
        ) -> None:
        """
        Initialize the NmfModel object w/ required data matrix and rank list.

        Parameters:
        - data: DataFrame on which NMF will be run
        - ranks: Iterable of ranks on which to perform NMF
        - max_iter: Integer, Max num of iters for convergence, default 10_000
        """
        # Check for NaN or infinite values in NMF input
        if data.isna().any().any() or np.isinf(data).any().any():
            raise ValueError("Input data contains NaN or infinite values")
        
        # Check for negative values in NMF input
        if (data < 0).any().any():
            raise ValueError("Input data contains negative values, which are not allowed in NMF")

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
        """Get input data for NMF models"""
        return self._data
    
    @property
    def ranks(self):
        """Get ranks on which NMF will be performed"""
        return self._ranks
    
    # If None, compute the following properties when called
    
    @property
    def W_dict(self):
        """Get a dictionary of raw W matrices across chosen ranks"""
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
        """Get a dictionary of raw H matrices across chosen ranks"""
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
        """Get a dictionary of L matrices across chosen ranks"""
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
        """Get a dictionary of A matrices across chosen ranks"""
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
        """Get a dictionary of binarized L matrices across chosen ranks"""
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
        """Get a dictionary of binarized A matrices across chosen ranks"""
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
        """
        Get a dictionary of the reconstructed data from post-processed NMF.
        """
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
        """
        Get a dictionary of errors between orig and reconstr data matrices.
        """
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
        """
        Get a dictionary of the confusion matrix.
        """
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
        """
        Return a table of metrics for NMF model reconstructions across ranks.
        """
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


# Container for PVGE models for easy loading into NmfData
class PVGE(object):
    """
    Polytope Vertex Group Extraction (PVGE) for clustering high-dimensional data.

    PVGE combines dimensionality reduction using DensMAP (a density-preserving variant
    of UMAP) with density-based clustering using HDBSCAN. This approach is particularly
    useful for identifying clusters in complex, high-dimensional genomic data.

    The process involves two main steps:
    1. Dimensionality Reduction: Using DensMAP to project the data into a lower-dimensional space
       while preserving both local and global structure.
    2. Clustering: Applying HDBSCAN to the reduced-dimensional data to identify clusters.

    Attributes:
    -----------
    data : pd.DataFrame
        The input high-dimensional data.
    densmap : UMAP
        The fitted DensMAP model.
    embedding : np.ndarray
        The low-dimensional embedding produced by DensMAP.
    hdbscan : HDBSCAN
        The fitted HDBSCAN model.
    labels : np.ndarray
        Cluster labels assigned by HDBSCAN.

    Methods:
    --------
    run_densmap()
        Perform DensMAP dimensionality reduction.
    run_hdbscan()
        Perform HDBSCAN clustering on the DensMAP embedding.

    Notes:
    ------
    DensMAP extends UMAP by incorporating density preservation into the optimization
    objective. This helps in maintaining the global density structure of the data
    in the low-dimensional embedding.

    HDBSCAN is a density-based clustering algorithm that extends DBSCAN by extracting
    a flat clustering based on the stability of clusters.

    References:
    -----------
    1. McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation
       and Projection for Dimension Reduction. ArXiv e-prints.
    2. Narayan, A., Berger, B., & Cho, H. (2021). Density-Preserving Data Visualization
       Unveils Dynamic Patterns of Single-Cell Transcriptomic Variability. Nature Biotechnology, 39, 765-774.
    3. Campello, R. J., Moulavi, D., & Sander, J. (2013). Density-based clustering based on
       hierarchical density estimates. In Pacific-Asia conference on knowledge discovery
       and data mining (pp. 160-172). Springer, Berlin, Heidelberg.
    """
    def __init__(
            self,
            data: pd.DataFrame,
            low_memory: bool = False,
            n_neighbors: int = None,
            max_range: int = None,
            core_dist_n_jobs: int = 8
    ) -> None:
        
        # data
        self._data = data

        # densmap
        self._densmap = None
        self._low_memory = low_memory
        
        if n_neighbors:
            self._n_neighbors = _check_n_neighbors(data, n_neighbors)
        else:
            self._n_neighbors = 0.01 * min(data.shape)
        
        # hdbscan
        self._hdbscan_best_model = None
        self._hdbscan_best_model_sil_score = None
        self._hdbscan_tuning_metrics = None
        self._max_range = max_range
        self._core_dist_n_jobs = core_dist_n_jobs
        self._labels = None
    
    # Properties
    @property
    def data(self):
        return self._data
    
    @property
    def densmap(self):
        if not self._densmap:
            self._densmap, self._embedding = run_densmap(
                self._data,
                self._low_memory,
                self._n_neighbors
            )
        
        return self._densmap
    
    @property
    def embedding(self):
        if not self._embedding:
            self._densmap, self._embedding = run_densmap(
                self._data,
                self._low_memory,
                self._n_neighbors
            )
        
        return self._embedding
    
    @property
    def n_neighbors(self):
        return self._n_neighbors
    
    @property
    def hdbscan(self):
        if not self._hdbscan_best_model:
            best_model, best_labels, best_model_sil_score, models_df = run_hdbscan(
                self._embedding,
                self._max_range,
                self._core_dist_n_jobs
            )
            self._hdbscan_best_model = best_model
            self._labels = best_labels
            self._hdbscan_best_model_sil_score = best_model_sil_score
            self._hdbscan_tuning_metrics = models_df
        
        return self._hdbscan_best_model
    
    @property
    def labels(self):
        if not self._labels:
            best_model, best_labels, best_model_sil_score, models_df = run_hdbscan(
                self._embedding,
                self._max_range,
                self._core_dist_n_jobs
            )
            self._hdbscan_best_model = best_model
            self._labels = best_labels
            self._hdbscan_best_model_sil_score = best_model_sil_score
            self._hdbscan_tuning_metrics = models_df
        
        return self._labels
    
    @property
    def silhouette_score(self):
        if not self._hdbscan_best_model_sil_score:
            best_model, best_labels, best_model_sil_score, models_df = run_hdbscan(
                self._embedding,
                self._max_range,
                self._core_dist_n_jobs
            )
            self._hdbscan_best_model = best_model
            self._labels = best_labels
            self._hdbscan_best_model_sil_score = best_model_sil_score
            self._hdbscan_tuning_metrics = models_df
        
        return self._hdbscan_best_model_sil_score
    
    @property
    def hdbscan_tuning_metrics(self):
        if not self._hdbscan_tuning_metrics:
            best_model, best_labels, best_model_sil_score, models_df = run_hdbscan(
                self._embedding,
                self._max_range,
                self._core_dist_n_jobs
            )
            self._hdbscan_best_model = best_model
            self._labels = best_labels
            self._hdbscan_best_model_sil_score = best_model_sil_score
            self._hdbscan_tuning_metrics = models_df
        
        return self._hdbscan_tuning_metrics
    
    # Class methods
    def run_densmap(self, low_memory: bool = False):
        logger.debug(f"Running DensMAP with n_neighbors={self._n_neighbors}")
        if low_memory:
            self._low_memory = low_memory
        
        densmap, embedding = run_densmap(
            self._data,
            self._low_memory,
            self._n_neighbors
        )
        self._densmap = densmap
        self._embedding = embedding
    
    def run_hdbscan(self, max_range: int = None, core_dist_n_jobs: int = None):
        if max_range:
            self._max_range = max_range
        if core_dist_n_jobs:
            self._core_dist_n_jobs = core_dist_n_jobs
        
        logger.debug(f"Running HDBSCAN with max_range={self._max_range}, "
                     f"core_dist_n_jobs={self._core_dist_n_jobs}")
        best_model, best_labels, best_model_sil_score, models_df = run_hdbscan(
            self.embedding,
            self._max_range,
            self._core_dist_n_jobs
        )
        self._hdbscan_best_model = best_model
        self._labels = best_labels
        self._hdbscan_best_model_sil_score = best_model_sil_score
        self._hdbscan_tuning_metrics = models_df


# Helper functions
def _k_means_binarize_L(L_norm):
    """
    Use k-means clustering (k=3) to binarize L_norm matrix.
    """
    
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
    """
    Use k-means clustering (k=3) to binarize A_norm matrix.
    """
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

def _calculate_metrics(P_confusion, P_reconstructed, rank):
    
    # Unpack confusion matrix elements
    TP = P_confusion[0, 0]
    FN = P_confusion[0, 1]
    FP = P_confusion[1, 0]
    TN = P_confusion[1, 1]
    
    # Use float for calculations to prevent integer overflow
    TP, FN, FP, TN = map(float, [TP, FN, FP, TN])
    Total = TP + TN + FP + FN
    
    # Calculations
    Precision = TP / (TP + FP) if TP + FP != 0 else 0
    Recall = TP / (TP + FN) if TP + FN != 0 else 0
    P_plus_R = Precision + Recall
    FPR = FP / (FP + TN) if FP + TN != 0 else 0
    FNR = FN / (TP + FN) if TP + FN != 0 else 0
    Specificity = TN / (TN + FP) if TN + FP != 0 else 0
    Prevalence = (TP + FN) / Total
    Accuracy = (TP + TN) / Total
    F1_score = 2 * (Precision * Recall) / (P_plus_R) if P_plus_R != 0 else 0
    BM = Recall + Specificity - 1 # a.k.a Youden's J statistic
    Jaccard_index = TP / (TP + FP + FN) if TP + FP + FN != 0 else 0

    # Adjusted MCC calculation to avoid overflow
    numerator = TP * TN - FP * FN
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    MCC = numerator / denominator if denominator != 0 else 0

    # Adjusted Prevalence Threshold to avoid overflow
    one_minus_Sp = 1 - Specificity
    if BM != 0:
        PT = (np.sqrt(Recall * one_minus_Sp) + Specificity - 1) / (BM)
    else:
        PT = 0

    # Calculate Akaike Information Criterion (AIC)
    Reconstruction_error = 1 - Jaccard_index # Jaccard distance (proxy for reconstr error)
    k = 2 * rank * (P_reconstructed.shape[0] + P_reconstructed.shape[1])  # number of parameters in NMF (W & H matrices)
    AIC = 2 * k + 2 * Reconstruction_error * Total
    
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
        'Jaccard Index': Jaccard_index,
        'AIC': AIC
    }

def _check_n_neighbors(data, n_neighbors):
    max_n = int(0.5 * min(data.shape))

    if n_neighbors > max_n:
        raise ValueError(
            f"n_neighbors is set too high at {n_neighbors},"
            "max allowed is {max_n} based on data shape of {data.shape})"
        )
    
    return n_neighbors

def recommended_threshold(A_norm, i):
    column_data_reshaped = A_norm.loc[f'phylon{i}'].values.reshape(-1, 1)
    
    # 3-means clustering
    kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto')
    kmeans.fit(column_data_reshaped)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    # Find the cluster with the highest mean
    highest_mean_cluster = np.argmax(centers)
    
    # Binarize the row based on the cluster with the highest mean
    binarized_row = (labels == highest_mean_cluster).astype(int)
    
    # Find k-means-recommended threshold using min value that still binarizes to 1
    x = pd.Series(dict(zip(A_norm.columns, binarized_row)))
    threshold = A_norm.loc[f'phylon{i}', x[x==1].index].min()
    
    return threshold
