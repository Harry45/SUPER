"""
Project: Scalable Gaussian Process Emulator (SUPER) for modelling power spectra
Authors: Rory Allen, Arrykrishna Mootoovaloo
"""

from typing import Tuple
from dataclasses import dataclass, field
import numpy as np
from classy import Class  # pylint: disable-msg=E0611

# our scripts and functions
import config as CONFIG


def class_compute(cosmology: dict):
    """Pre-computes the quantities in CLASS.
    Args:
        cosmology (dict): A dictionary with the cosmological parameters
    Returns:
        module: A CLASS module
    """

    # instantiate Class
    class_module = Class()

    # set cosmology
    class_module.set(CONFIG.CLASS_ARGS)

    # flat universe
    class_module.set({'Omega_k': CONFIG.OMEGA_K})

    # configuration for neutrino
    class_module.set(CONFIG.NEUTRINO_SETTINGS)

    # BBN prediction of the primordial Helium abundance
    class_module.set({'k_pivot': CONFIG.K_PIVOT})
    class_module.set({'sBBN file': CONFIG.BBN})

    if CONFIG.NEUTRINO:
        class_module.set({'m_ncdm': cosmology['M_tot'] / CONFIG.NEUTRINO_SETTINGS['deg_ncdm']})

    else:
        class_module.set({'m_ncdm': CONFIG.FIXED_NM['M_tot'] / CONFIG.NEUTRINO_SETTINGS['deg_ncdm']})

    # set basic configurations for Class
    cosmo = {k: cosmology[k] for k in CONFIG.PARAMS}

    class_module.set(cosmo)

    # compute the important quantities
    class_module.compute()

    return class_module


def delete_module(module):
    """Deletes the module to prevent memory overflow.
    Args:
        module: A CLASS module
    """
    module.struct_cleanup()

    module.empty()

    del module


@dataclass
class PowerSpectrum:
    """Calculates the linear matter power spectrum using CLASS
    Args:
        z_min (float): the minimum redshift
        z_max (float): the maximum redshift
        k_min (float): the minimum wavenumber [unit: 1/Mpc]
        k_max (float): the maximum wavenumber [unit: 1/Mpc]
    """

    # the range of redshifts and wavenumbers to consider
    z_min: float = field(default=0.0)
    z_max: float = field(default=5.0)
    k_min: float = field(default=1e-4, metadata={"unit": "h Mpc^-1"})
    k_max: float = field(default=1.0, metadata={"unit": "h Mpc^-1"})

    def __post_init__(self):
        self.wavenumber = np.geomspace(
            self.k_min, self.k_max, CONFIG.NWAVE, endpoint=True
        )

    def pk_calculation(self, cosmology: dict, redshift: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates the linear matter power spectrum at a fixed redshift.
        Args:
            cosmology (dict): A dictionary of values following CLASS notation,
            for example, cosmology = {'Omega_b': 0.022, 'Omega_cdm': 0.12,
            'n_s': 1.0, 'h':0.75, 'ln10^{10}A_s': 3.0}
            redshift (float, optional): The redshift at which the power spectrum
            is computed. Defaults to 0.0.
        Returns:
            np.ndarray: The linear matter power spectrum
        """

        # compute the power spectrum
        class_module = class_compute(cosmology)

        # create an empty list to store the linear matter spectrum
        pk_linear = np.zeros_like(self.wavenumber)

        # for the non linear matter power spectrum
        pk_nonlinear = np.zeros_like(self.wavenumber)

        for i, wav in enumerate(self.wavenumber):

            # get the power spectrum
            pk_linear[i] = class_module.pk_lin(wav, redshift)

            # non linear matter power spectrum
            pk_nonlinear[i] = class_module.pk(wav, redshift)

        # delete the CLASS module
        delete_module(class_module)

        return pk_linear, pk_nonlinear
