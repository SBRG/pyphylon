import os, shutil, urllib.request, urllib.parse, urllib.error
import subprocess as sp
import hashlib 
import collections

import pandas as pd
import numpy as np
import scipy.sparse

from tqdm.notebook import tqdm

from pangenome.pangenome import *

CLUSTER_TYPES = {'cds':'C', 'noncoding':'T'}
VARIANT_TYPES = {'allele':'A', 'upstream':'U', 'downstream':'D'}
CLUSTER_TYPES_REV = {v:k for k,v in list(CLUSTER_TYPES.items())}
VARIANT_TYPES_REV = {v:k for k,v in list(VARIANT_TYPES.items())}
DNA_COMPLEMENT = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 
              'W': 'W', 'S': 'S', 'R': 'Y', 'Y': 'R', 
              'M': 'K', 'K': 'M', 'N': 'N'}
for bp in list(DNA_COMPLEMENT.keys()):
    DNA_COMPLEMENT[bp.lower()] = DNA_COMPLEMENT[bp].lower()


def cds_pangenome_update(original_pangenome_path, genome_faa_paths, output_dir, name='Test', 
                        cdhit_args={'-n':5, '-c':0.8}, 
                        fastasort_path=None, save_csv=True):
    ''' 
    // ADD IN DESCRIPTION OF THE OUTPUT AND PROCESS THE FUNCTION GOES THROUGH
    
    Parameters
    ----------
    original_pangenome_path: str
        Path to the .faa file for the original pangenome the new strains are being added to
    genome_faa_paths : list 
        FAA files containing CDSs for genomes of interest. Genome 
        names are inferred from these FAA file paths.
    output_dir : str
        Path to directory to generate outputs and intermediates.
    name : str
        Header to prepend to all output files and allele names (default 'Test')
    cdhit_args : dict
        Alignment arguments to pass CD-Hit, other than -i, -o, and -d
        (default {'-n':5, '-c':0.8})
    fastasort_path : str
        Path to Exonerate's fastasort binary, optionally for sorting
        final FAA files (default None)
    save_csv : bool
        If true, saves allele and gene tables as csv.gz. May be limiting
        step for very large tables (default True)
        
    Returns 
    -------
    df_alleles : pd.DataFrame
        Binary allele x genome table
    df_genes : pd.DataFrame
        Binary gene x genome table
    '''
    
    """
    TODO:
    1. Find the max number of clusters in the pangenome to set the last cluster
    2. Find all allels from the new pangenome which are in a cluster with a known allele (and if they have 100% similarity)
        - This appears to be the most complicated part to make work, need to think about how CD-HIT finds its clusters (best hit or first hit)
        - May need to find a way to match for best allele post cd-hit in order to see if I need to make a new allele name or not
    3. Name alleles which matched existing clusters to correct name (with allele information from last step)
    4. For novel sequences, name them based on the number of already existing clusters in the pangenome
    5. Make P and P_allele matrices here for the new sequences (decide if novel sequences should be included here or not)
    6. Make header to allele file for new sequences
    7. Return all relevant structures and output any necessary files to the disk
    """
    

    return df_alleles, df_genes, df_alleles_new, df_genes_new, header_to_allele # may need to change what we are returning here, not sure
