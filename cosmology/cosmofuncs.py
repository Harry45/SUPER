"""
Project: Scalable Gaussian Process Emulator (SUPER) for modelling power spectra
Authors: Rory Allen, Arrykrishna Mootoovaloo
"""
import numpy as np
import config as CONFIG


def analytical_baryon_feedback(wavenumber: np.ndarray, redshift: np.ndarray, amplitude: float) -> np.ndarray:
    """Fitting formula for baryon feedback following equation 10 and Table 2
    from J. Harnois-Deraps et al. 2014 (arXiv.1407.4301)

    Args:
        wavenumber (np.ndarray): the wavenumber in h/Mpc
        redshift (np.ndarray): the redshift
        amplitude (float): the amplitude of the baryon feedback

    Returns:
        np.ndarray: the bias squred term, b^2(k,z)
    """
    wavenumber = np.atleast_2d(wavenumber).T

    redshift = np.atleast_2d(redshift)

    model = CONFIG.BARYON_MODEL

    # k is expected in h/Mpc and is divided in log by this unit...
    x_wav = np.log10(wavenumber)

    # calculate  the scale factor, a
    a_factor = 1. / (1. + redshift)

    # a squared
    a_sqr = a_factor * a_factor

    a_z = CONFIG.CST[model]['A2'] * a_sqr + CONFIG.CST[model]['A1'] * a_factor + CONFIG.CST[model]['A0']
    b_z = CONFIG.CST[model]['B2'] * a_sqr + CONFIG.CST[model]['B1'] * a_factor + CONFIG.CST[model]['B0']
    c_z = CONFIG.CST[model]['C2'] * a_sqr + CONFIG.CST[model]['C1'] * a_factor + CONFIG.CST[model]['C0']
    d_z = CONFIG.CST[model]['D2'] * a_sqr + CONFIG.CST[model]['D1'] * a_factor + CONFIG.CST[model]['D0']
    e_z = CONFIG.CST[model]['E2'] * a_sqr + CONFIG.CST[model]['E1'] * a_factor + CONFIG.CST[model]['E0']

    # original formula:
    # bias_sqr = 1.-A_z*np.exp((B_z-C_z)**3)+D_z*x*np.exp(E_z*x)
    # original formula with a free amplitude A_bary:
    bias_sqr = 1. - amplitude * (a_z * np.exp((b_z * x_wav - c_z)**3) - d_z * x_wav * np.exp(e_z * x_wav))

    return bias_sqr
