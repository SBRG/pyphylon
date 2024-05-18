"""
Functions for downloading genomes.
"""

import os
import ftplib
from Bio import Entrez

# Directory locations
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'metadata')
GENOME_SUMMARY_DIR = os.path.join(DATA_DIR, 'genome_summary')
GENOME_METADATA_DIR = os.path.join(DATA_DIR, 'genome_metadata')
PATRIC_GENOME_AMR_DIR = os.path.join(DATA_DIR, 'PATRIC_genome_AMR.txt')


# TODO: Add in checks from 1b and deduplication
# TODO: Add in functions to download selected strains

def download_bvbrc_genome_info_files(force=False):
    """
    Download genome summary, genome metadata, and PATRIC_genome_AMR files
    from BV-BRC. If files already exist, they will not be downloaded unless
    force=True.
    
    Parameters:
    - force (bool): Boolean indicating whether to force re-download of files.
    """
    _ensure_directories_exist()
    
    files_to_download = {
        'genome_summary': 'RELEASE_NOTES/genome_summary',
        'genome_metadata': 'RELEASE_NOTES/genome_metadata',
        'PATRIC_genome_AMR': 'RELEASE_NOTES/PATRIC_genome_AMR.txt'
    }

    for file_name, ftp_path in files_to_download.items():
        local_path = os.path.join(DATA_DIR, file_name)
        if not os.path.exists(local_path) or force:
            print(f"Downloading {file_name} from {ftp_path}...")
            download_from_bvbrc(ftp_path, local_path)
        else:
            print(f"{file_name} already exists. Skipping download. Use force=True to re-download.")


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

# Helper function
def _ensure_directories_exist():
    """
    Ensure that necessary directories exist.
    """
    os.makedirs(GENOME_SUMMARY_DIR, exist_ok=True)
    os.makedirs(GENOME_METADATA_DIR, exist_ok=True)
    os.makedirs(PATRIC_GENOME_AMR_DIR, exist_ok=True)
