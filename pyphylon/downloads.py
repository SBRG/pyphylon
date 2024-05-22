"""
Functions for downloading genomes.
"""

import os
import ftplib
import logging
import time
import requests

from Bio import Entrez
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
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

    # Set up Selenium WebDriver
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run headless
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)
    driver.get(url)

    # Allow time for JavaScript to execute
    time.sleep(5)  # Adjust as needed for the page to load completely

    # Extract page source and parse with BeautifulSoup
    page_source = driver.page_source
    driver.quit()

    soup = BeautifulSoup(page_source, 'html.parser')
    reference_genome_link = None
    
    # Find the <a> tag with '/datasets/genome' in its href attribute
    for link in soup.find_all('a', href=True):
        if '/datasets/genome/GC' in link['href']:
            reference_genome_link = link['href']
            logging.info(f"Found reference genome link: {reference_genome_link}")
            break
    
    if reference_genome_link is None:
        raise ValueError(f"Reference genome link not found for taxon ID {taxon_id}")
    
    return reference_genome_link

def get_scaffold_n50(reference_genome_url):
    """
    Retrieves the Scaffold N50 value from the reference genome page using Selenium.
    
    Parameters:
    reference_genome_url (str): The URL of the reference genome page.
    
    Returns:
    int: The Scaffold N50 value in base units.
    """
    # Set up Selenium WebDriver
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run headless
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)
    driver.get(reference_genome_url)

    # Allow time for JavaScript to execute
    time.sleep(5)  # Adjust as needed for the page to load completely

    # Extract page source and parse with BeautifulSoup
    page_source = driver.page_source
    driver.quit()

    soup = BeautifulSoup(page_source, 'html.parser')
    scaffold_n50 = None
    
    # Find the <td> element containing the text "Scaffold N50" and the next <td> element
    scaffold_n50_td = soup.find('td', text="Scaffold N50")
    if scaffold_n50_td:
        logging.info("Found 'Scaffold N50' cell.")
        next_td = scaffold_n50_td.find_next('td')
        if next_td:
            scaffold_n50 = next_td.text.strip()
            logging.info(f"Found Scaffold N50 value: {scaffold_n50}")
        else:
            logging.warning(f"No following <td> element found for 'Scaffold N50'.")
    else:
        logging.warning(f"'Scaffold N50' cell not found in the table.")
    
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
