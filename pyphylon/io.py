"""
Functions for reading and writing data into files.
"""

import json
import joblib
from typing import Any

from .core import NmfData
from .models import NmfModel


def save_nmf_data(nmf_data: NmfData, filepath: str, **kwargs):
    """
    Save an NmfData object to a file.

    Parameters:
    - nmf_data (NmfData): The NmfData object to save.
    - filepath (str): The path to the file where the object will be saved.
    - kwargs: Additional kwargs to pass onto joblib.dump()
    """
    processed_filepath = filepath if '.pkl' in filepath[-4:] else f'{filepath}.pkl'
    joblib.dump(nmf_data, processed_filepath, **kwargs)


def load_nmf_data(filepath: str) -> NmfData:
    """
    Load an NmfData object from a file.

    Parameters:
    - filepath (str): The path to the file from which to load the object.

    Returns:
    - NmfData: The loaded NmfData object.
    """
    data = joblib.load(filepath)
    return NmfData(data)


def save_nmf_model(nmf_model: NmfModel, filepath: str, **kwargs):
    """
    Save an NmfModel object to a file.

    Parameters:
    - nmf_model (NmfModel): The NmfModel object to save.
    - filepath (str): The path to the file where the object will be saved.
    - kwargs: Additional kwargs to pass onto joblib.dump()
    """
    processed_filepath = filepath if '.pkl' in filepath[-4:] else f'{filepath}.pkl'
    joblib.dump(nmf_model, processed_filepath, **kwargs)


def load_nmf_model(filepath: str) -> NmfModel:
    """
    Load an NmfModel object from a file.

    Parameters:
    - filepath (str): The path to the file from which to load the object.

    Returns:
    - NmfModel: The loaded NmfModel object.
    """
    data = joblib.load(filepath)
    return NmfModel(data)


def save_nmf_data_to_json(nmf_data: NmfData, filepath: str):
    """
    Save an NmfData object to a JSON file.

    Parameters:
    - nmf_data (NmfData): The NmfData object to save.
    - filepath (str): The path to the file where the object will be saved.
    """
    data_dict = nmf_data.__dict__
    with open(filepath, 'w') as f:
        json.dump(data_dict, f)


def load_nmf_data_from_json(filepath: str) -> NmfData:
    """
    Load an NmfData object from a JSON file.

    Parameters:
    - filepath (str): The path to the file from which to load the object.

    Returns:
    - NmfData: The loaded NmfData object.
    """
    with open(filepath, 'r') as f:
        data_dict = json.load(f)
    nmf_data = NmfData(**data_dict)
    return nmf_data


def save_nmf_model_to_json(nmf_model: NmfModel, filepath: str):
    """
    Save an NmfModel object to a JSON file.

    Parameters:
    - nmf_model (NmfModel): The NmfModel object to save.
    - filepath (str): The path to the file where the object will be saved.
    """
    model_dict = nmf_model.__dict__
    with open(filepath, 'w') as f:
        json.dump(model_dict, f)


def load_nmf_model_from_json(filepath: str) -> NmfModel:
    """
    Load an NmfModel object from a JSON file.

    Parameters:
    - filepath (str): The path to the file from which to load the object.

    Returns:
    - NmfModel: The loaded NmfModel object.
    """
    with open(filepath, 'r') as f:
        model_dict = json.load(f)
    nmf_model = NmfModel(**model_dict)
    return nmf_model
