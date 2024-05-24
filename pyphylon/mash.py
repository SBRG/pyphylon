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

def mash_sketch_genomes(genome_dir, output_file):
    """
    Generate a Mash sketch file.

    Parameters:
    - genome_dir (str): Path to genome files for Mash analysis
    - output_file (str): Path for output file. If non
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
