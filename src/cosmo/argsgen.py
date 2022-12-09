"""
Project: Scalable Gaussian Process Emulator (SUPER) for modelling power spectra
Author: Dr. Arrykrishna Mootoovaloo
Date: November 2022
Email: arrykrish@gmail.com
Description: Generate the dictionaries to be passed to Class to compute the linear/non-linear matter power spectrum.
"""


from ml_collections.config_dict import ConfigDict


def class_args(config: ConfigDict) -> dict:
    """Generates CLASS arguments to be passed to classy to compute the different quantities.

    Args:
        config (ConfigDict): A configuration file containing the parameters.

    Returns:
        dict: A dictionary to input to class
    """
    dictionary = dict()
    dictionary['output'] = config.classy.output
    dictionary['non_linear'] = config.classy.mode
    dictionary['P_k_max_1/Mpc'] = config.classy.k_max_pk
    dictionary['z_max_pk'] = config.classy.z_max_pk
    dictionary['sBBN file'] = config.classy.bbn
    dictionary['k_pivot'] = config.classy.k_pivot
    dictionary['Omega_k'] = config.classy.Omega_k
    return dictionary


def neutrino_args(config: ConfigDict, **kwargs) -> dict:
    """Generates a dictionary for the neutrino settings.

    Args:
        config (ConfigDict): The main configuration file

    Returns:
        dict: A dictionary with the neutrino parameters.
    """
    dictionary = dict()
    dictionary['N_ncdm'] = config.neutrino.N_ncdm
    dictionary['deg_ncdm'] = config.neutrino.deg_ncdm
    dictionary['T_ncdm'] = config.neutrino.T_ncdm
    dictionary['N_ur'] = config.neutrino.N_ur

    if 'M_tot' in kwargs and 'M_tot' in config.parameters.names:
        mass = kwargs.pop('M_tot')
    else:
        mass = config.neutrino.fixed_nm
    dictionary['m_ncdm'] = mass / config.neutrino.deg_ncdm
    return dictionary


def params_args(config: ConfigDict, values: dict) -> dict:
    """Generates a dictionary of parameters, which will be used as input to Classy

    Args:
        config (ConfigDict): the main configuration file
        values (dict): the values of the parameters

    Returns:
        dict: a dictionary, with the approximate keys and values
    """
    assert len(values) == len(config.parameters.names), 'Mis-match in shape'

    dictionary = dict()
    for name in config.parameters.names:
        if name not in ['M_tot']:
            dictionary[name] = values[name]

    if 'c_min' not in config.parameters.names:
        dictionary['c_min'] = config.classy.cmin

    return dictionary
