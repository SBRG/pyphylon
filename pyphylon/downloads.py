"""
Functions for the downloading genomes.
"""

import os
import ftplib

# Directory locations
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
GENOME_SUMMARY_DIR = os.path.join(DATA_DIR, 'genome_summary')
GENOME_METADATA_DIR = os.path.join(DATA_DIR, 'genome_metadata')

# Download site-wide BV-BRC genome files
def download_bvbrc_genome_files(force=False):
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

# Function to download data from BV-BRC via FTP
def download_from_bvbrc(ftp_url, save_path):
    with ftplib.FTP('ftp.bvbrc.org') as ftp:
        ftp.login()
        with open(save_path, 'wb') as f:
            ftp.retrbinary(f'RETR ' + ftp_url, f.write)

# Function to download data from NCBI using Entrez
def download_from_ncbi(query, save_path, email='your_email@example.com'):
    Entrez.email = email
    handle = Entrez.esearch(db='genome', term=query)
    record = Entrez.read(handle)
    ids = record['IdList']
    if ids:
        handle = Entrez.efetch(db='genome', id=ids[0], rettype='gb', retmode='text')
        with open(save_path, 'w') as f:
            f.write(handle.read())