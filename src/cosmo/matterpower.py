"""
Authors: Arrykrishna Mootoovaloo
Email: arrykrish@gmail.com
Date: November 2022
Project: Implementation of a scalable GP approach for emulating power spectra
Script: Script for generating the linear and non-linear matter power spectrum.
"""

from ml_collections.config_dict import ConfigDict
from classy import Class  # pylint: disable-msg=E0611
import numpy as np

# our scripts and functions
from utils.logger import get_logger
from .argsgen import class_args, neutrino_args, params_args
from .cosmofuncs import sigma_eight


def class_compute(config: ConfigDict, cosmology: dict) -> Class:
    """Pre-computes the quantities in CLASS.
    Args:
        config (ConfigDict): The main configuration file for running Class
        cosmology (dict): A dictionary with the cosmological parameters
    Returns:
        Class: A CLASS module
    """
    # generates the dictionaries to input to Class
    arg_class = class_args(config)

    if config.boolean.neutrino:
        nu_mass = cosmology['M_tot']
        arg_neutrino = neutrino_args(config, M_tot=nu_mass)
    else:
        arg_neutrino = neutrino_args(config)

    arg_params = params_args(config, cosmology)

    if 'S_8' in config.parameters.names:
        arg_params['sigma8'] = sigma_eight(arg_neutrino | arg_params)
        del arg_params['S_8']

    # logger = get_logger(config, 'class')
    # logger.info('Running Class at %s', cosmology)

    # Run Class
    class_module = Class()
    class_module.set(arg_class)
    class_module.set(arg_neutrino)
    class_module.set(arg_params)
    class_module.compute()

    return class_module


def delete_module(class_module: Class):
    """Deletes the module to prevent memory overflow.
    Args:
        module (Class): A CLASS module
    """
    class_module.struct_cleanup()

    class_module.empty()

    del class_module


def calculate_pk_fixed_redshift(config: ConfigDict, cosmology: dict, redshift: float = 0) -> np.ndarray:
    """Calculates the linear or non-linear matter power spectrum

    Args:
        config (ConfigDict): the set of configurations for running CLASS
        cosmology (dict): a dictionary consisting of the cosmological parameters
        redshift (float): the redshift at which we want to calculate the power spectrum

    Returns:
        np.ndarray: the power spectrum at a fixed redshift
    """
    wavenumbers = np.geomspace(config.emulator.kmin, config.emulator.kmax, config.emulator.grid_nk)
    powerspec = np.zeros(config.emulator.grid_nk)
    module = class_compute(config, cosmology)

    for i, wav in enumerate(wavenumbers):
        if config.boolean.linearpk:
            powerspec[i] = module.pk_lin(wav, redshift)
        else:
            powerspec[i] = module.pk(wav, redshift)
    delete_module(module)
    return powerspec


def calculate_pk(config: ConfigDict, cosmology: dict) -> np.ndarray:
    """Calculates the linear or non-linear matter power spectrum

    Args:
        config (ConfigDict): the set of configurations for running CLASS
        cosmology (dict): a dictionary consisting of the cosmological parameters

    Returns:
        np.ndarray: the power spectrum of size Nk x Nz, based on the number of values of k and z in the configuration.
    """
    module = class_compute(config, cosmology)
    wavenumbers = np.geomspace(config.emulator.kmin, config.emulator.kmax, config.emulator.grid_nk)
    redshifts = np.linspace(config.emulator.zmin, config.emulator.zmax, config.emulator.grid_nz)
    powerspec = np.zeros((config.emulator.grid_nk, config.emulator.grid_nz))
    for i in range(config.emulator.grid_nk):
        for j in range(config.emulator.grid_nz):
            if config.boolean.linearpk:
                powerspec[i, j] = module.pk_lin(wavenumbers[i], redshifts[j])
            else:
                powerspec[i, j] = module.pk(wavenumbers[i], redshifts[j])
    delete_module(module)
    return powerspec
