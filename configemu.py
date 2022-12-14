"""
Authors: Arrykrishna Mootoovaloo
Email: arrykrish@gmail.com
Date: November 2022
Project: Implementation of a scalable GP approach for emulating power spectra
Script: The main configuration file
"""
from ml_collections.config_dict import ConfigDict

BF_PARAMS = {'AGN': {'A2': -0.11900, 'B2': 0.1300, 'C2': 0.6000, 'D2': 0.002110, 'E2': -2.0600,
                     'A1': 0.30800, 'B1': -0.6600, 'C1': -0.7600, 'D1': -0.002950, 'E1': 1.8400,
                     'A0': 0.15000, 'B0': 1.2200, 'C0': 1.3800, 'D0': 0.001300, 'E0': 3.5700},
             'REF': {'A2': -0.05880, 'B2': -0.2510, 'C2': -0.9340, 'D2': -0.004540, 'E2': 0.8580,
                     'A1': 0.07280, 'B1': 0.0381, 'C1': 1.0600, 'D1': 0.006520, 'E1': -1.7900,
                     'A0': 0.00972, 'B0': 1.1200, 'C0': 0.7500, 'D0': -0.000196, 'E0': 4.5400},
             'DBLIM': {'A2': -0.29500, 'B2': -0.9890, 'C2': -0.0143, 'D2': 0.001990, 'E2': -0.8250,
                       'A1': 0.49000, 'B1': 0.6420, 'C1': -0.0594, 'D1': -0.002350, 'E1': -0.0611,
                       'A0': -0.01660, 'B0': 1.0500, 'C0': 1.3000, 'D0': 0.001200, 'E0': 4.4800}}


def get_config() -> ConfigDict:
    """Generates the main configuration file for Class and the emulator. Note that the wavenumber is in inverse Mpc.

    Returns:
        ConfigDict: the configuration file
    """

    config = ConfigDict()
    config.logname = 'Scalable-GP'

    # boolean settings
    config.boolean = boolean = ConfigDict()
    boolean.neutrino = False
    boolean.baryonfeedback = False
    boolean.fixedbf = False
    boolean.bf_analytic = False
    boolean.linearpk = True

    # paths
    config.path = path = ConfigDict()
    path.data = 'data/'
    path.gps = 'gps/' + 'linear/' if boolean.linearpk else 'nonlinear/'
    path.plots = 'plots/'
    path.logs = 'logs/'

    # parameters
    config.parameters = parameters = ConfigDict()
    parameters.names = ['omega_cdm', 'omega_b', 'S_8', 'n_s', 'h']
    parameters.distribution = 'uniform'
    parameters.loc = [0.051, 0.019, 0.10, 0.84, 0.64]
    parameters.scale = [0.204, 0.007, 1.20, 0.26, 0.18]
    parameters.fiducial = [0.12, 0.020, 0.76, 1.0, 0.70]
    parameters.nparams = len(parameters.names)

    # settings for analytical baryon feedback model
    config.bar_fed = bar_fed = ConfigDict()
    bar_fed.model = 'AGN'
    bar_fed.params = BF_PARAMS[config.bar_fed.model]

    # CLASS settings
    config.classy = classy = ConfigDict()
    classy.halofit_k_per_decade = 80
    classy.halofit_sigma_precision = 0.05
    classy.output = "mPk"
    classy.mode = 'hmcode'
    classy.cmin = 3.13
    # classy.bbn = '/home/harry/Desktop/class/bbn/sBBN.dat'
    classy.bbn = '/home/mootovaloo/Desktop/class/external/bbn/sBBN.dat'
    classy.k_pivot = 0.05
    classy.Omega_k = 0.0
    classy.k_max_pk = 50
    classy.z_max_pk = 2.0

    # neutrino settings
    config.neutrino = neutrino = ConfigDict()
    neutrino.N_ncdm = 1.0
    neutrino.deg_ncdm = 3.0
    neutrino.T_ncdm = 0.71611
    neutrino.N_ur = 0.00641
    neutrino.fixed_nm = 0.06

    # emulator settings
    config.emulator = emulator = ConfigDict()
    emulator.niter = 500
    emulator.lr = 1E-2
    emulator.nrestart = 3
    emulator.zmin = 0.0
    emulator.zmax = classy.z_max_pk
    emulator.kmin = 5E-4
    emulator.kmax = classy.k_max_pk
    emulator.grid_nz = 20
    emulator.grid_nk = 40

    # spline interpolator settings
    config.interp = interp = ConfigDict()
    interp.gridz = 100
    interp.gridk = 1000

    return config
