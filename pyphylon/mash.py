"""
Functions for running Mash analysis.
"""

import logging
import subprocess

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hc
import scipy.spatial as sp

from kneebow.rotor import Rotor

def sketch_genomes(genome_dir, output_file):
    """
    Generate a Mash sketch file.

    Parameters:
    - genome_dir (str): Path to genome files for Mash analysis
    - output_file (str): Path for output Mash sketch file.
    """
    # Configure logging
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Ensure all ".fna" files in genome_dir are selected for
    if not output_file.endswith('/*.fna'):
        if output_file.endswith('/'):
            output_file += "*.fna"
        else:
            output_file += "/*.fna"

    # Generate command
    cmd = [
        "mash", "sketch",
        "-o", f"{output_file}",
        f"{genome_dir}"
    ]

    # Run command
    result = subprocess.run(
        cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    return result

def generate_pairwise_distances(mash_sketch_file, output_file):
    """
    Generate a table of all pairwise Mash distances.

    Parameters:
    - mash_sketch_file (str): Path to Mash sketch file
    - output_file (str): Path for output Mash distances table.
    """
    # Generate command
    cmd = [
        "mash", "dist",
        f"{mash_sketch_file}", "{mash_sketch_file}", ">", f"{output_file}"
    ]

    # Run command
    result = subprocess.run(
        cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    return result

def cluster_corr_dist(df_mash_corr_dist, thresh=0.1, method='ward', metric='euclidean'):
    '''
    Hierarchically Mash-based pairwise-pearson-distance matrix
    '''
    link = hc.linkage(sp.distance.squareform(df_mash_corr_dist), method=method, metric=metric)
    dist = sp.distance.squareform(df_mash_corr_dist)
    
    clst = pd.DataFrame(index=df_mash_corr_dist.index)
    clst['cluster'] = hc.fcluster(link, thresh * dist.max(), 'distance')
    
    return link, dist, clst


def remove_bad_strains(df_mash_scd, bad_strains_list):
    good_strains_list = sorted(set(df_mash_scd.index) - set(bad_strains_list))
    
    return df_mash_scd.loc[good_strains_list, good_strains_list]


# Sensitivity analysis to pick the threshold (for E. coli we use 0.1)
# We pick the threshold where the curve just starts to bottom out
def sensitivity_analysis(df_mash_corr_dist_complete):
    x = list(np.logspace(-3, -1, 10)) + list(np.linspace(0.1, 1, 19))
    
    def num_uniq_clusters(thresh):
        link = hc.linkage(sp.distance.squareform(df_mash_corr_dist_complete), method='ward', metric='euclidean')
        dist = sp.distance.squareform(df_mash_corr_dist_complete)
        
        clst = pd.DataFrame(index=df_mash_corr_dist_complete.index)
        clst['cluster'] = hc.fcluster(link, thresh * dist.max(), 'distance')
        
        return len(clst.cluster.unique())
    
    tmp = pd.DataFrame()
    tmp['threshold'] = pd.Series(x)
    tmp['num_clusters'] = pd.Series(x).apply(num_uniq_clusters)
    
    # Find which value the elbow corresponds to
    df_temp = tmp.sort_values(by='num_clusters', ascending=True).reset_index(drop=True)
    
    # transform input into form necessary for package
    results_itr = zip(list(df_temp.index), list(df_temp.num_clusters))
    data = list(results_itr)
    
    rotor = Rotor()
    rotor.fit_rotate(data)
    elbow_idx = rotor.get_elbow_index()
    df_temp['num_clusters'][elbow_idx]
    contamination_cutoff = df_temp['num_clusters'][elbow_idx]
    
    # Grab elbow threshold
    cond = tmp['num_clusters'] == df_temp['num_clusters'][elbow_idx]
    elbow_threshold = tmp[cond]['threshold'].iloc[0]
    
    return tmp, df_temp, elbow_idx, elbow_threshold


