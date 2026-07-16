"""
Functions for downloading genomes.
"""

import os
import ftplib
import logging
import shutil
import time
import requests
import urllib
import pandas as pd

from typing import Union
from Bio import Entrez
from tqdm.notebook import tqdm


# URLs
GENOME_SUMMARY_URL = "https://zenodo.org/record/11226678/files/genome_summary_Oct_12_23.tsv?download=1"
GENOME_METADATA_URL = "https://zenodo.org/record/11226678/files/genome_metadata_Oct_12_23.tsv?download=1"

# Valid filetype options for BV-BRC genome sequence downloads
VALID_BV_BRC_FILES = ['faa','features.tab','ffn','frn','gff','pathway.tab', 'spgene.tab','subsystem.tab','fna']

# BV-BRC HTTPS Data API (replaces FTP which now requires SSL/TLS on control channel)
BV_BRC_API_BASE = "https://www.bv-brc.org/api"
BV_BRC_API_ENDPOINTS = {
    'fna': ('genome_sequence', 'application/sralign+dna+fasta'),
    'gff': ('genome_feature', 'application/gff'),
    'faa': ('genome_feature', 'application/protein+fasta'),
    'ffn': ('genome_feature', 'application/dna+fasta'),
    'frn': ('genome_feature', 'application/dna+fasta'),
    'features.tab': ('genome_feature', 'text/tsv'),
    'pathway.tab': ('pathway', 'text/tsv'),
    'spgene.tab': ('sp_gene', 'text/tsv'),
    'subsystem.tab': ('subsystem', 'text/tsv'),
}



# TODO: Add in checks from 1b and deduplication
# TODO: Add in functions to download from NCBI (including RefSeq)

# Genome info downloads
def download_bvbrc_genome_info(output_dir=None, force=False):
    """
    Download genome summary, genome metadata, and PATRIC_genome_AMR files
    from BV-BRC. If files already exist, they will not be downloaded unless
    force=True.
    
    Parameters:
    - output_dir (str): Directory to save the downloaded files. Defaults to the curr work dir if None.
    - force (bool): Boolean indicating whether to force re-download of files.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Set the default output directory to the current working directory if output_dir is None
    if output_dir is None:
        output_dir = os.getcwd()
    else:
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    files_to_download = {
        'genome_summary.tsv': 'RELEASE_NOTES/genome_summary',
        'genome_metadata.tsv': 'RELEASE_NOTES/genome_metadata',
        'PATRIC_genome_AMR.tsv': 'RELEASE_NOTES/PATRIC_genome_AMR.txt'
    }

    for file_name, ftp_path in files_to_download.items():
        local_path = os.path.join(output_dir, file_name)
        if not os.path.exists(local_path) or force:
            logging.info(f"Downloading {file_name} from {ftp_path}...")
            download_from_bvbrc(ftp_path, local_path)
        else:
            logging.info(f"{file_name} already exists. Skipping download. Use force=True to re-download.")
            continue

def download_example_bvbrc_genome_info(output_dir=None, force=False):
    """
    Downloads example genome metadata and summary files (Oct 12 2023) from Zenodo.
    
    Parameters:
    - output_dir (str): Directory to save the downloaded files. Defaults to the curr work dir if None.
    - force (bool): Force download even if the file already exists. Defaults to False.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Set the default output directory to the current working directory if output_dir is None
    if output_dir is None:
        output_dir = os.getcwd()
    else:
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    # Define the URLs for the files to be downloaded
    urls = {
        "genome_metadata_Oct_12_23.tsv": GENOME_METADATA_URL,
        "genome_summary_Oct_12_23.tsv": GENOME_SUMMARY_URL
    }
    
    # Download each file
    logging.info("Starting download...")
    for filename, url in urls.items():
        file_path = os.path.join(output_dir, filename)
        if not force and os.path.exists(file_path):
            logging.info(f"File {filename} already exists in {output_dir} and force is set to False. Skipping download.")
            continue
        
        try:
            response = requests.get(url)
            response.raise_for_status()  # Check if the request was successful
            with open(file_path, 'wb') as file:
                file.write(response.content)
            logging.info(f"Downloaded {filename} to {file_path}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download {filename}: {e}")


def query_bvbrc_genomes(taxon_id, genome_status=None, genome_quality=None, limit=25000):
    """
    Query BV-BRC API for genomes by taxon ID (includes all descendant taxa).

    Parameters:
    - taxon_id (int/str): NCBI taxonomy ID (e.g., 197 for C. jejuni)
    - genome_status (str, optional): Filter by status ('Complete', 'WGS')
    - genome_quality (str, optional): Filter by quality ('Good', 'Fair', 'Poor')
    - limit (int): Max records per request (default 25000)

    Returns:
    - pd.DataFrame: Genome records from BV-BRC
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    base_url = "https://www.bv-brc.org/api/genome/"

    # Build RQL query — use taxon_lineage_ids to include subspecies/strains
    rql_parts = [f"eq(taxon_lineage_ids,{taxon_id})"]
    if genome_status is not None:
        rql_parts.append(f"eq(genome_status,{genome_status})")
    if genome_quality is not None:
        rql_parts.append(f"eq(genome_quality,{genome_quality})")

    all_records = []
    offset = 0

    while True:
        rql_query = "&".join(rql_parts + [f"limit({limit},{offset})"])
        url = f"{base_url}?{rql_query}"
        headers = {"Accept": "application/json"}

        logging.info(f"Querying BV-BRC API (offset={offset})...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        records = response.json()
        if not records:
            break

        all_records.extend(records)
        logging.info(f"Retrieved {len(records)} records (total: {len(all_records)})")

        if len(records) < limit:
            break
        offset += limit

    if not all_records:
        logging.warning(f"No genomes found for taxon_id={taxon_id}")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    logging.info(f"Total genomes retrieved: {df.shape[0]}")
    return df


# Genome sequence downloads
def download_genome_sequences(df_or_filepath: Union[str, pd.DataFrame], output_dir, force=False):
    """
    Download .fna and .gff files for strains from a DataFrame.

    Parameters:
    - df_or_filepath (str or DataFrame): DataFrame or filepath of the DataFrame.
    - output_dir (str): Directory to save the downloaded files.
    - force (bool): Boolean indicating whether to force re-download of files.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load the DataFrame directly or from a provided path
    if isinstance(df_or_filepath, pd.DataFrame):
        df = df_or_filepath.copy()
    elif str(df_or_filepath).endswith('.csv'):
        logging.info(f"Loading CSV file {df_or_filepath} to DataFrame...")
        df = pd.read_csv(df_or_filepath, index_col=0, dtype='object')
    elif str(df_or_filepath).endswith('.pickle') or str(df_or_filepath).endswith('.pickle.gz'):
        logging.info(f"Loading DATAFRAME PICKLE file {df_or_filepath} to DataFrame...")
        df = pd.read_pickle(df_or_filepath)
    else:
        raise TypeError(f"df_or_filepath must be a DataFrame or a str/filepath. An object of type {type(df_or_filepath)} was passed instead.")
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Make subfolders for fna and gff files
    fna_subdir = os.path.join(output_dir, 'fna')
    gff_subdir = os.path.join(output_dir, 'gff')

    # Ensure the output subdirectories exist
    if not os.path.exists(fna_subdir):
        os.makedirs(fna_subdir)
    if not os.path.exists(gff_subdir):
        os.makedirs(gff_subdir)

    # Iterate through the strains in the DataFrame
    logging.info(f"Downloading genomes from BV-BRC...")
    for genome_id in tqdm(df['genome_id'].astype('str')):
        
        # Construct FTP paths for .fna and .gff files
        fna_ftp_path = f"genomes/{genome_id}/{genome_id}.fna"
        gff_ftp_path = f"genomes/{genome_id}/{genome_id}.PATRIC.gff"
        
        # Construct local save paths
        fna_save_path = os.path.join(fna_subdir, f"{genome_id}.fna")
        gff_save_path = os.path.join(gff_subdir, f"{genome_id}.gff")
        
        # Download the .fna file
        download_from_bvbrc(fna_ftp_path, fna_save_path, force)
        
        # Download the .gff file
        download_from_bvbrc(gff_ftp_path, gff_save_path, force)

def download_genomes_bvbrc(genomes, output_dir, filetypes=['fna','gff'], force=False):
    '''
    Download data associated with a list of PATRIC genomes.
    
    Parameters:
    - genomes (list): List of strings containing PATRIC genome IDs to download
    - output_dir (str): Path to directory to save genomes. Will create a subfolder for each filetype in filetypes
    - filetypes (list): List of BV-BRC genome-specific files to download per genome.
        Valid options include 'faa', 'features.tab', 'ffn', 'frn',
        'gff', 'pathway.tab', 'spgene.tab', 'subsystem.tab', and
        'fna'. 'PATRIC' in filename is dropped automatically.
        See ftp://ftp.bvbrc.org/genomes/<genome id>/ for
        examples (default ['fna','gff'])
    - force (bool): If True, re-downloads files that exist locally (default False)
    Returns:
    - bad_genomes (list): List of PATRIC genome IDs that could not be downloaded
    '''
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Initialize list of "bad" genomes that failed to download
    bad_genomes = []
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # Initialize vars prior to looping
    source_target_filetypes = []
    subdir = dict.fromkeys(filetypes)

    # Process filetypes
    for ftype in tqdm(filetypes, desc='Processing filetypes...'):

        # Make subfolders for each ftype
        subdir[ftype] = os.path.join(output_dir, ftype)
        if not os.path.exists(subdir[ftype]):
            os.makedirs(subdir[ftype])

        # Check if ftype is a valid BV-BRC filetype
        if ftype in VALID_BV_BRC_FILES:
            # all files except FNA preceded by "PATRIC"
            ftype_source = f"PATRIC.{ftype}" if ftype != "fna" else ftype
            # drop "PATRIC" in output files
            ftype_target = ftype
            source_target_filetypes.append( (ftype_source, ftype_target) )
        
        # Check if ftype without PATRIC label is a valid BV-BRC filetype
        elif ftype.replace('PATRIC.','') in VALID_BV_BRC_FILES:
            # keep "PATRIC" for downloading files
            ftype_source = ftype
            # drop "PATRIC" in output files
            ftype_target = ftype.replace('PATRIC.','')
            source_target_filetypes.append( (ftype_source, ftype_target) )
        
        # Invalid filetype
        else:
            logging.info(f"Invalid filetype: {ftype}")
            continue
    
    # Download relevant files via BV-BRC HTTPS Data API
    for genome in tqdm(genomes, desc='Downloading selected files...', total=len(genomes)):
        for source_filetype, target_filetype in source_target_filetypes:
            target = os.path.join(subdir[target_filetype], f"{genome}.{target_filetype}")

            if os.path.exists(target) and not force:
                logging.info(f"File {target} already exists and force is False. Skipping download.")
                continue

            # Look up the API endpoint and accept header for this filetype
            api_resource, accept_header = BV_BRC_API_ENDPOINTS[target_filetype]
            url = (
                f"{BV_BRC_API_BASE}/{api_resource}/"
                f"?eq(genome_id,{genome})&http_accept={accept_header}&limit(25000)"
            )

            logging.info(f"Downloading {target_filetype} for {genome} via HTTPS API...")
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req) as response:
                    data = response.read()
                if not data:
                    logging.warning(f"Empty response for {target_filetype} of genome {genome}")
                    bad_genomes.append(genome)
                    continue
                with open(target, 'wb') as f:
                    f.write(data)
            except (urllib.error.HTTPError, urllib.error.URLError) as e:
                logging.warning(f"Failed to download {target_filetype} for genome {genome}: {e}")
                if os.path.exists(target):
                    os.remove(target)
                bad_genomes.append(genome)

    # Remove related "bad" genome files:
    for bad_genome in tqdm(bad_genomes, desc='Removing bad genome files...'):
        for ftype, subdir_path in subdir.items():
            bad_genome_path = os.path.join(subdir_path, f"{bad_genome}.{ftype}")
            if os.path.exists(bad_genome_path):
                os.remove(bad_genome_path)
    
    # Return a list of bad genomes that failed to download
    return bad_genomes

# Retrieval functions
def get_scaffold_n50_for_species(taxon_id):
    """
    Retrieves the Scaffold N50 value for a species' reference genome via NCBI Datasets API.

    Parameters:
    - taxon_id (str/int): NCBI taxonomy ID (e.g., 197 for C. jejuni, 562 for E. coli)

    Returns:
    - int: Scaffold N50 in base pairs
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    url = f"https://api.ncbi.nlm.nih.gov/datasets/v2/genome/taxon/{taxon_id}/dataset_report"
    params = {"filters.reference_only": "true", "page_size": 1}

    logging.info(f"Querying NCBI Datasets API for reference genome (taxon_id={taxon_id})...")
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    reports = data.get("reports", [])
    if not reports:
        raise ValueError(f"No reference genome found for taxon ID {taxon_id}")

    scaffold_n50 = reports[0]["assembly_stats"]["scaffold_n50"]
    logging.info(f"Scaffold N50 for taxon {taxon_id}: {scaffold_n50}")
    return int(scaffold_n50)

# Helper functions
def download_from_bvbrc(ftp_path, save_path, force=False):
    """
    Download a file from BV-BRC FTP server.

    Parameters:
    - ftp_path (str): The FTP path of the file to download.
    - save_path (str): The local path to save the downloaded file.
    - force (bool): Boolean indicating whether to force re-download of files.
    """
    if not force and os.path.exists(save_path):
        logging.info(f"File {save_path} already exists & force is set to False. Skipping download.")
        return

    with ftplib.FTP('ftp.bv-brc.org') as ftp:
        ftp.login()
        with open(save_path, 'wb') as f:
            ftp.retrbinary(f'RETR {ftp_path}', f.write)

def download_from_ncbi(query, save_path, email='your_email@example.com'):
    """
    Download genome data from NCBI using Entrez.

    Parameters:
    - query (str): Query string for searching NCBI.
    - save_path (str): Local path where the file will be saved.
    - email (str): User's email address for Entrez.

    Returns:
    - None
    """
    Entrez.email = email
    handle = Entrez.esearch(db='genome', term=query)
    record = Entrez.read(handle)
    ids = record['IdList']
    if ids:
        handle = Entrez.efetch(db='genome', id=ids[0], rettype='gb', retmode='text')
        with open(save_path, 'w') as f:
            f.write(handle.read())

