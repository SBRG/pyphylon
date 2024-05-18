"""
Functions for reading and writing data into files.
"""

import json
import pickle
import pandas as pd

from .core import NmfData
from .models import NmfModel

def save_nmf_data(nmf_data: NmfData, filepath):
    """
    Save an NmfData object to a file.

    Parameters:
    - nmf_data (NmfData): The NmfData object to save.
    - filepath (str): The path to the file where the object will be saved.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(nmf_data, f)

def load_nmf_data(filepath):
    """
    Load an NmfData object from a file.

    Parameters:
    - filepath (str): The path to the file from which to load the object.

    Returns:
    - NmfData: The loaded NmfData object.
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_nmf_model(nmf_model: NmfModel, filepath):
    """
    Save an NmfModel object to a file.

    Parameters:
    - nmf_model (NmfModel): The NmfModel object to save.
    - filepath (str): The path to the file where the object will be saved.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(nmf_model, f)

def load_nmf_model(filepath):
    """
    Load an NmfModel object from a file.

    Parameters:
    - filepath (str): The path to the file from which to load the object.

    Returns:
    - NmfModel: The loaded NmfModel object.
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_json(data, filepath):
    """
    Save data to a JSON file.

    Parameters:
    - data (dict): The data to save.
    - filepath (str): The path to the file where the data will be saved.
    """
    with open(filepath, 'w') as f:
        json.dump(data, f)

def load_json(filepath):
    """
    Load data from a JSON file.

    Parameters:
    - filepath (str): The path to the file from which to load the data.

    Returns:
    - dict: The loaded data.
    """
    with open(filepath, 'r') as f:
        return json.load(f)
