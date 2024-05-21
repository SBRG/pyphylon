"""
Functions for downloading genomes.
"""

import os
import ftplib
import requests
import logging
from Bio import Entrez

# URLs
GENOME_SUMMARY_URL = "https://zenodo.org/record/11226678/files/genome_summary_Oct_12_23.tsv?download=1"
GENOME_METADATA_URL = "https://zenodo.org/record/11226678/files/genome_metadata_Oct_12_23.tsv?download=1"
# PATRIC_GENOME_AMR_URL = os.path.join(DATA_DIR, 'PATRIC_genome_AMR.txt')


# TODO: Add in checks from 1b and deduplication
# TODO: Add in functions to download selected strains
# TODO: Add in functions to download from NCBI (including RefSeq)

def download_bvbrc_genome_info_files(output_dir=None, force=False):
    """
    Download genome summary, genome metadata, and PATRIC_genome_AMR files
    from BV-BRC. If files already exist, they will not be downloaded unless
    force=True.
    
    Parameters:
    - output_dir (str): Directory to save the downloaded files. Defaults to the current working directory if None.
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

def download_example_bvbrc_genome_files(output_dir=None, force=False):
    """
    Downloads genome metadata and summary files from Zenodo.
    
    Parameters:
    - output_dir (str): Directory to save the downloaded files. Defaults to the current working directory if None.
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

# Example usage from 1a.ipynb:
# To download to the metadata folder
download_example_bvbrc_genome_files(output_dir='data/metadata')

# To download to the current working directory
download_example_bvbrc_genome_files()

# To force download to the metadata folder
download_example_bvbrc_genome_files(output_dir='data/metadata', force=True)


def download_from_bvbrc(ftp_path, save_path):
    """
    Download a file from the BV-BRC FTP server.

    Parameters:
    - ftp_path (str): Path to the file on the FTP server.
    - save_path (str): Local path where the file will be saved.
    """
    with ftplib.FTP('ftp.bvbrc.org') as ftp:
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
