"""
Functions for downloading genomes.
"""

import os
import ftplib
import requests
import logging
from Bio import Entrez
from bs4 import BeautifulSoup


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

def get_scaffold_n50_for_species(taxon_id):
    """
    Retrieves the Scaffold N50 value for a given species by its taxon ID.
    
    Parameters:
    taxon_id (str): The taxon ID of the species.
    
    Returns:
    int: The Scaffold N50 value in base units.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(f"Fetching reference genome link for taxon ID {taxon_id}")
    reference_genome_url = get_reference_genome_link(taxon_id)
    full_reference_genome_url = f"https://www.ncbi.nlm.nih.gov{reference_genome_url}"
    logging.info(f"Fetching Scaffold N50 value from {full_reference_genome_url}")
    scaffold_n50 = get_scaffold_n50(full_reference_genome_url)

    return scaffold_n50

# Helper functions
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

def get_reference_genome_link(taxon_id):
    """
    Retrieves the reference genome link from NCBI for a given taxon ID.
    
    Parameters:
    taxon_id (str): The taxon ID of the species.
    
    Returns:
    str: The URL of the reference genome page.
    """
    url = f"https://www.ncbi.nlm.nih.gov/datasets/taxonomy/{taxon_id}"
    response = requests.get(url)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.content, 'html.parser')
    reference_genome_link = None
    
    # Find the <h3> tag with the class and text "Reference genome"
    reference_genome_link_tag = soup.find(
        'a',
        class_='MuiTypography-root MuiTypography-inherit MuiLink-root MuiLink-underlineHover css-m18yf3'
    )
    if reference_genome_link_tag:
        logging.info(f"Found reference genome link: {reference_genome_link_tag['href']}")
        reference_genome_link = reference_genome_link_tag['href']
    else:
        logging.warning(f"Reference genome link not found for taxon ID {taxon_id}.")
    
    if reference_genome_link is None:
        raise ValueError(f"Reference genome link not found for taxon ID {taxon_id}")
    
    return reference_genome_link

def get_scaffold_n50(reference_genome_url):
    """
    Retrieves the Scaffold N50 value from the reference genome page.
    
    Parameters:
    reference_genome_url (str): The URL of the reference genome page.
    
    Returns:
    int: The Scaffold N50 value in base units.
    """
    response = requests.get(reference_genome_url)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.content, 'html.parser')
    scaffold_n50 = None
    
    # Find the Scaffold N50 value under the RefSeq column
    assembly_statistics_heading = soup.find(text="Assembly statistics")
    if assembly_statistics_heading:
        logging.info("Found 'Assembly statistics' heading.")
        table = assembly_statistics_heading.find_next('table')
        if table:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 2 and "Scaffold N50" in cells[0].text:
                    scaffold_n50 = cells[1].text.strip()
                    logging.info(f"Found Scaffold N50 value: {scaffold_n50}")
                    break
    else:
        logging.warning(f"'Assembly statistics' heading not found at {reference_genome_url}.")
    
    if scaffold_n50 is None:
        raise ValueError(f"Scaffold N50 value not found at {reference_genome_url}")
    
    return _convert_to_int(scaffold_n50)

def _convert_to_int(value_str):
    """
    Converts a string with units to an integer.
    
    Parameters:
    value_str (str): The string containing the numeric value and units (e.g., "5.3 Mb").
    
    Returns:
    int: The numeric value in base units.
    """
    units = {'Kb': 1e3, 'Mb': 1e6, 'Gb': 1e9}
    value, unit = value_str.split()
    return int(float(value) * units[unit])
