"""
Authors: Arrykrishna Mootoovaloo
Email: arrykrish@gmail.com
Date: November 2022
Project: Implementation of a scalable GP approach for emulating power spectra
Script: Generates the training points for the emulator.
"""
import os
from ml_collections.config_dict import ConfigDict
import scipy.stats
import pandas as pd
import logging

# our scripts and functions
from utils.helpers import save_csv, save_pickle, load_pickle  # pylint: disable=import-error
from utils.logger import get_logger
from src.cosmo.matterpower import calculate_pk

logger = logging.getLogger(__name__)


def generate_prior(config: ConfigDict) -> dict:
    """Generates the entity of each parameter by using scipy.stats function.

    Args:
        dictionary (dict): A dictionary with the specifications of the prior.

    Returns:
        dict: the prior distribution of all parameters.
    """
    dictionary = dict()
    for i, key in enumerate(config.parameters.names):
        specs = (config.parameters.loc[i], config.parameters.scale[i])
        dictionary[key] = getattr(scipy.stats, config.parameters.distribution)(*specs)
    return dictionary


def scale_lhs(config: ConfigDict, lhs_file: str, save: bool = False, **kwargs) -> list:
    """Scale the Latin Hypercube Samples according to the prior range.

    Args:
        config (ConfigDict): the main configuration file
        lhs_file (str): The name of the file
        save (bool): Whether to save the scaled LHS samples. Defaults to False.

    Returns:
        list: A list of dictionaries containing the scaled LHS samples (cosmological parameters).
    """
    logger.info("Scaling the LHS samples to the prior range.")

    # read the LHS samples
    lhs = pd.read_csv(os.path.join("data", lhs_file + ".csv"), index_col=[0])
    priors = generate_prior(config)
    cosmo_list = list()

    for i in range(lhs.shape[0]):

        lhs_row = lhs.iloc[i, :].values  # pylint: disable=maybe-no-member
        cosmo = dict()

        for k in range(config.parameters.nparams):
            param = config.parameters.names[k]
            cosmo[param] = round(priors[param].ppf(lhs_row[k]), 4)

        # append to the list
        cosmo_list.append(cosmo)

    if save:
        fname = kwargs.pop('fname')
        cosmos_df = pd.DataFrame(cosmo_list)
        save_csv(cosmos_df, "data", "cosmologies_" + fname)
        save_pickle(cosmo_list, "data", "cosmologies_" + fname)

    return cosmo_list


def generate_training_pk(config: ConfigDict, fname: str) -> dict:
    """Generates the training set for the power spectra

    Args:
        config (ConfigDict): the main configuration file.
        fname (str): name of the file.

    Returns:
        dict: a dictionary with calculated power spectra (2D surface)
    """
    logger.info("Generating the training set for the power spectra.")

    cosmologies = load_pickle('data', 'cosmologies_' + fname)
    ncosmo = len(cosmologies)
    record_pk = dict()
    for i in range(ncosmo):
        record_pk[i] = calculate_pk(config, cosmologies[i])
    file_name = 'pk_' + 'lin' if config.boolean.linearpk else 'nonlin'
    save_pickle(record_pk, 'data', file_name + f'_{ncosmo}')
    return record_pk
