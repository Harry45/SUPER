"""
Generates the training points: cosmological parameters (inputs) and linear matter power
spectrum (targets).

Project: Scalable Gaussian Process Emulator(SUPER) for modelling power spectra
Authors: Rory Allen, Arrykrishna Mootoovaloo
"""
import os
from typing import Tuple
import scipy.stats
import pandas as pd

# our scripts and functions
from src.powerspectrum import PowerSpectrum
import utils.helpers as hp
import config as CONFIG


def generate_prior(dictionary: dict) -> dict:
    """Generates the entity of each parameter by using scipy.stats function.

    Args:
        dictionary (dict): A dictionary with the specifications of the prior.

    Returns:
        dict: the prior distribution of the parameter.
    """
    dist = getattr(scipy.stats, dictionary["distribution"])(*dictionary["specs"])

    return dist


def scale_lhs(fname: str = "lhs_500", save: bool = True) -> list:
    """Scale the Latin Hypercube Samples according to the prior range.

    Args:
        fname (str, optional): The name of the LHS file. Defaults to 'lhs_500'.
        save (bool): Whether to save the scaled LHS samples. Defaults to True.

    Returns:
        list: A list of dictionaries with the scaled LHS samples.
    """

    # read the LHS samples
    path = os.path.join("data", fname + ".csv")

    lhs = pd.read_csv(path, index_col=[0])

    # number of training points
    ncosmo = lhs.shape[0]

    # create an empty list to store the cosmologies
    cosmo_list = list()

    # create an empty list to store the distributions
    priors = {}

    for param in CONFIG.PARAMS:
        priors[param] = generate_prior(CONFIG.PRIORS[param])

    for i in range(ncosmo):
        # get the cosmological parameters
        cosmo = lhs.iloc[i, :]

        # scale the cosmological parameters
        cosmo = {
            CONFIG.PARAMS[k]: priors[CONFIG.PARAMS[k]].ppf(cosmo[k])
            for k in range(len(CONFIG.PARAMS))
        }

        # append to the list
        cosmo_list.append(cosmo)

    if save:
        cosmos_df = pd.DataFrame(cosmo_list)
        hp.save_csv(cosmos_df, "data", "cosmologies_" + fname)
        hp.save_list(cosmo_list, "data", "cosmologies_" + fname)

    return cosmo_list


def calculate_power_spectrum(fname: str, redshift: float = 0.0) -> Tuple[list, list, list]:
    """Generates the linear matter power spectrum.

    Args:
        fname (str): The name of the file.
        redshift (float): The redshift. Defaults to 0.0.

    Returns:
        Tuple[list, list, list]: A list of the cosmological parameters and
        correponding list of the linear and nonlinear matter power spectrum.
    """

    # scale the LHS points to the cosmological parameters
    cosmo_params = scale_lhs(fname, save=True)

    # class to compute the linear matter power spectrum
    module = PowerSpectrum(CONFIG.ZMIN, CONFIG.ZMAX, CONFIG.KMIN, CONFIG.KMAX)

    pk_lin = list()
    pk_non = list()

    for cosmo in cosmo_params:

        pk_linear, pk_nonlinear = module.pk_calculation(cosmo, redshift)

        # record the linear and non-linear matter power spectrum
        pk_lin.append(pk_linear)
        pk_non.append(pk_nonlinear)

    pk_lin_df = pd.DataFrame(pk_lin)
    pk_non_df = pd.DataFrame(pk_non)

    # save the linear matter power spectrum
    hp.save_csv(pk_lin_df, "data", "pk_linear_" + fname)
    hp.save_list(pk_lin, "data", "pk_linear_" + fname)

    # save the non linear matter power spectrum
    hp.save_csv(pk_non_df, "data", "pk_nonlinear_" + fname)
    hp.save_list(pk_non, "data", "pk_nonlinear_" + fname)

    return cosmo_params, pk_lin, pk_non
