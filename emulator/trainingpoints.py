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

# our scripts and functions
from utils.helpers import save_csv, save_pickle  # pylint: disable=import-error


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

    # read the LHS samples
    lhs = pd.read_csv(os.path.join("data", lhs_file + ".csv"), index_col=[0])
    priors = generate_prior(config)
    cosmo_list = list()

    for i in range(lhs.shape[0]):
        # get the cosmological parameters
        lhs_row = lhs.iloc[i, :].values
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


# def calculate_power_spectrum(fname: str, redshift: float = 0.0) -> Tuple[list, list, list]:
#     """Generates the linear matter power spectrum.

#     Args:
#         fname (str): The name of the file.
#         redshift (float): The redshift. Defaults to 0.0.

#     Returns:
#         Tuple[list, list, list]: A list of the cosmological parameters and
#         correponding list of the linear and nonlinear matter power spectrum.
#     """

#     # scale the LHS points to the cosmological parameters
#     cosmo_params = scale_lhs(fname, save=True)

#     # class to compute the linear matter power spectrum
#     module = PowerSpectrum(CONFIG.ZMIN, CONFIG.ZMAX, CONFIG.KMIN, CONFIG.KMAX)

#     pk_lin = list()
#     pk_non = list()

#     for cosmo in cosmo_params:

#         pk_linear, pk_nonlinear = module.pk_calculation(cosmo, redshift)

#         # record the linear and non-linear matter power spectrum
#         pk_lin.append(pk_linear)
#         pk_non.append(pk_nonlinear)

#     pk_lin_df = pd.DataFrame(pk_lin)
#     pk_non_df = pd.DataFrame(pk_non)

#     # save the linear matter power spectrum
#     hp.save_csv(pk_lin_df, "data", "pk_linear_" + fname)
#     hp.save_list(pk_lin, "data", "pk_linear_" + fname)

#     # save the non linear matter power spectrum
#     hp.save_csv(pk_non_df, "data", "pk_nonlinear_" + fname)
#     hp.save_list(pk_non, "data", "pk_nonlinear_" + fname)

#     return cosmo_params, pk_lin, pk_non
