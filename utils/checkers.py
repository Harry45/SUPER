"""
Authors: Arrykrishna Mootoovaloo
Email: arrykrish@gmail.com
Date: November 2022
Project: Implementation of a scalable GP approach for emulating power spectra
Script: Check that all conditions are satisfied
"""
import os
from ml_collections.config_dict import ConfigDict


def check_config(config: ConfigDict) -> None:
    """Check if all boolean conditions are satisfied in the config file.

    Args:
        config (ConfigDict): the main configuration file.
    """

    if config.boolean.neutrino:
        assert 'M_tot' in config.parameters.names, 'Missing parameter name for neutrino'


def make_paths(config: ConfigDict) -> None:
    """Make sure all relevant folders where we store outputs exist.

    Args:
        config (ConfigDict): the main configuration file.
    """
    os.makedirs(config.path.data, exist_ok=True)
    os.makedirs(config.path.gps, exist_ok=True)
    os.makedirs(config.path.plots, exist_ok=True)
