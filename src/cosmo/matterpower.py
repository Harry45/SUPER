"""
Authors: Arrykrishna Mootoovaloo
Email: arrykrish@gmail.com
Date: November 2022
Project: Implementation of a scalable GP approach for emulating power spectra
Script: Script for generating the linear and non-linear matter power spectrum.
"""

import logging
from typing import Tuple
from dataclasses import dataclass, field
from ml_collections.config_dict import ConfigDict
import numpy as np
from classy import Class  # pylint: disable-msg=E0611

# our scripts and functions
from utils.logger import get_logger
from .argsgen import class_args, neutrino_args, params_args


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

    logger = get_logger(config, 'class')
    logger.info('Running Class')

    # Run Class
    class_module = Class()
    class_module.set(arg_class)
    class_module.set(arg_neutrino)
    class_module.set(arg_params)
    class_module.compute()

    return class_module


def delete_module(class_module):
    """Deletes the module to prevent memory overflow.
    Args:
        module: A CLASS module
    """
    class_module.struct_cleanup()

    class_module.empty()

    del class_module
