# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: This file contains some helper functions.

import os
import pickle
import numpy as np
import pandas as pd
import torch


def tensor_to_dict(tensor: torch.tensor, keys: list) -> dict:
    """Convert a tensor to a dictionary given a list of keys

    Args:
        tensor (torch.tensor): The tensor to convert.
        keys (list): The list of keys.

    Returns:
        dict: A dictionary with the keys and values of the tensor.
    """
    return {key: tensor[i].item() for i, key in enumerate(keys)}


def dict_to_tensor(dictionary: dict, keys: list) -> torch.tensor:
    """Converts a dictionary to a tensor.

    Args:
        dictionary (dict): the dictionary to convert
        keys (list): the list of keys (usually in the setting file)

    Returns:
        torch.tensor: the pytorch tensor
    """

    return torch.tensor([dictionary[key] for key in keys])


def subset_dict(dictionary: dict, keys: list) -> dict:
    """Generates a subset of a dictionary.

    Args:
        dictionary (dict): A long dictionary with keys and values respectively.
        keys (list): A list of keys to be extracted.

    Returns:
        dict: A dictionary with only the keys specified.
    """

    return {key: dictionary[key] for key in keys}


def store_arrays(array: np.ndarray, folder_name: str, file_name: str) -> None:
    """Stores a numpy array in a folder.

    Args:
        array (np.ndarray): The array to store.
        folder_name (str): The name of the folder.
        file_name (str): The name of the file.
    """

    # create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # use compressed format to store data
    np.savez_compressed(folder_name + "/" + file_name + ".npz", array)


def load_arrays(folder_name: str, file_name: str) -> np.ndarray:
    """Load the arrays from a folder.

    Args:
        folder_name (str): name of the folder.
        file_name (str): name of the file.

    Returns:
        np.ndarray: The array.
    """

    matrix = np.load(folder_name + "/" + file_name + ".npz")["arr_0"]

    return matrix


def load_csv(folder_name: str, file_name: str) -> pd.DataFrame:
    """Given a folder name and file name, we will load the csv file.

    Args:
        folder_name(str): the name of the folder
        file_name(str): name of the file

    Returns:
        pd.DataFrame: the loaded csv file
    """
    # load the csv file
    df = pd.read_csv(folder_name + "/" + file_name + ".csv", header=None)

    return df


def save_csv(array: np.ndarray, folder_name: str, file_name: str) -> None:
    """Save an array to a csv file

    Args:
        array (np.ndarray): The array to be saved
        folder_name (str): The name of the folder
        file_name (str): The name of the file
    """
    # create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    np.savetxt(folder_name + "/" + file_name + ".csv", array, delimiter=",")


def save_pd_csv(df: pd.DataFrame, folder_name: str, file_name: str) -> None:
    """Save an array to a csv file

    Args:
        array (np.ndarray): The array to be saved
        folder_name (str): The name of the folder
        file_name (str): The name of the file
    """
    # create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    df.to_csv(folder_name + "/" + file_name + ".csv", index=False)


def save_list(list_to_store: list, folder_name: str, file_name: str) -> None:
    """Stores a list in a folder.

    Args:
        list_to_store (list): The list to store.
        folder_name (str): The name of the folder.
        file_name (str): The name of the file.
    """

    # create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # use compressed format to store data
    with open(folder_name + "/" + file_name + ".pkl", "wb") as f:
        pickle.dump(list_to_store, f)


def load_list(folder_name: str, file_name: str) -> list:
    """Reads a list from a folder.

    Args:
        folder_name (str): The name of the folder.
        file_name (str): The name of the file.

    Returns:
        list: The list.
    """

    with open(folder_name + "/" + file_name + ".pkl", "rb") as f:
        list_to_read = pickle.load(f)

    return list_to_read


def emcee_chains(folder_name: str, file_name: str, save=True) -> np.ndarray:
    """Reads the emcee chains from a folder.

    Args:
        folder_name (str): The name of the folder.
        file_name (str): The name of the file.

    Returns:
        np.ndarray: The array.
    """

    # the pkl file
    pkl_file = load_list(folder_name, file_name)

    # the MCMC samples
    array = pkl_file.flatchain

    # number of samples
    nsamples = array.shape[0]

    # the log-posterior
    logp = pkl_file.flatlnprobability.reshape(nsamples, 1)

    # the samples are unique - so we add a column of ones in the beginning
    ones = np.ones((nsamples, 1))

    # combine the important information
    comb = np.concatenate([ones, logp, array], axis=1)

    # save the array if required
    if save:
        np.savetxt(folder_name + "/" + "MCE_" + file_name, comb)

    return comb
