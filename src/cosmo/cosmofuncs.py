"""
Authors: Arrykrishna Mootoovaloo
Email: arrykrish@gmail.com
Date: November 2022
Project: Implementation of a scalable GP approach for emulating power spectra
Script: Functions related to the cosmology, for example, analytical baryon feedback.
"""
import torch
from ml_collections.config_dict import ConfigDict

# our script and functions
from configemu import BF_PARAMS


def analytical_baryon_feedback(config: ConfigDict, wavenumber: torch.Tensor, redshift: torch.Tensor,
                               amplitude: torch.Tensor) -> torch.Tensor:
    """Fitting formula for baryon feedback following equation 10 and Table 2
    from J. Harnois-Deraps et al. 2014 (arXiv.1407.4301)

    Args:
        wavenumber (torch.Tensor): the wavenumber in h/Mpc
        redshift (torch.Tensor): the redshift
        amplitude (torch.Tensor): the amplitude of the baryon feedback

    Returns:
        torch.Tensor: the bias squred term, b^2(k,z)
    """
    wavenumber = wavenumber.view(-1, 1)
    redshift = redshift.view(1, -1)
    model = config.bar_fed.model

    # k is expected in h/Mpc
    x_wav = torch.log10(wavenumber)

    # calculate the scale factor, a
    a_factor = 1. / (1. + redshift)

    # a squared
    a_sqr = a_factor * a_factor

    a_z = BF_PARAMS[model]['A2'] * a_sqr + BF_PARAMS[model]['A1'] * a_factor + BF_PARAMS[model]['A0']
    b_z = BF_PARAMS[model]['B2'] * a_sqr + BF_PARAMS[model]['B1'] * a_factor + BF_PARAMS[model]['B0']
    c_z = BF_PARAMS[model]['C2'] * a_sqr + BF_PARAMS[model]['C1'] * a_factor + BF_PARAMS[model]['C0']
    d_z = BF_PARAMS[model]['D2'] * a_sqr + BF_PARAMS[model]['D1'] * a_factor + BF_PARAMS[model]['D0']
    e_z = BF_PARAMS[model]['E2'] * a_sqr + BF_PARAMS[model]['E1'] * a_factor + BF_PARAMS[model]['E0']

    # original formula:
    # bias_sqr = 1.-A_z*np.exp((B_z-C_z)**3)+D_z*x*np.exp(E_z*x)
    # original formula with a free amplitude A_bary:
    bias_sqr = 1. - amplitude * (a_z * torch.exp((b_z * x_wav - c_z)**3) - d_z * x_wav * torch.exp(e_z * x_wav))
    return bias_sqr
