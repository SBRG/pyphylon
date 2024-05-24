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
