"""
Project: Scalable Gaussian Process Emulator (SUPER) for modelling power spectra
Authors: Rory Allen, Arrykrishna Mootoovaloo
"""

# whether we want to sample the neutrino mass
NEUTRINO = True

# whether we want to include baryon feedback model in the pipeline
BARYON_FEEDBACK = True

# which Baryon Feedback model to use (halofit or hmcode)
MODE = 'hmcode'

# settings for neutrino
NEUTRINO_SETTINGS = {'N_ncdm': 1.0, 'deg_ncdm': 3.0, 'T_ncdm': 0.71611, 'N_ur': 0.00641}

if not NEUTRINO:

    # fixed neutrino mass
    FIXED_NM = {'M_tot': 0.06}

# settings for halofit

# halofit needs to evaluate integrals (linear power spectrum times some
# kernels). They are sampled using this logarithmic step size (default in CLASS is 80)

HALOFIT_K_PER_DECADE = 80.

# a smaller value will lead to a more precise halofit result at the
# highest redshift at which halofit can make computations,at the expense
# of requiring a larger k_max; but this parameter is not relevant for the
# precision on P_nl(k,z) at other redshifts, so there is normally no need
# to change it (default in CLASS is 0.05)

HALOFIT_SIGMA_PRECISION = 0.05

# -----------------------------------------------------------------------------
# Baryon Feedback settings

# baryon model to be used
BARYON_MODEL = 'AGN'

CST = {'AGN': {'A2': -0.11900, 'B2': 0.1300, 'C2': 0.6000, 'D2': 0.002110, 'E2': -2.0600,
               'A1': 0.30800, 'B1': -0.6600, 'C1': -0.7600, 'D1': -0.002950, 'E1': 1.8400,
               'A0': 0.15000, 'B0': 1.2200, 'C0': 1.3800, 'D0': 0.001300, 'E0': 3.5700},
       'REF': {'A2': -0.05880, 'B2': -0.2510, 'C2': -0.9340, 'D2': -0.004540, 'E2': 0.8580,
               'A1': 0.07280, 'B1': 0.0381, 'C1': 1.0600, 'D1': 0.006520, 'E1': -1.7900,
               'A0': 0.00972, 'B0': 1.1200, 'C0': 0.7500, 'D0': -0.000196, 'E0': 4.5400},
       'DBLIM': {'A2': -0.29500, 'B2': -0.9890, 'C2': -0.0143, 'D2': 0.001990, 'E2': -0.8250,
                 'A1': 0.49000, 'B1': 0.6420, 'C1': -0.0594, 'D1': -0.002350, 'E1': -0.0611,
                 'A0': -0.01660, 'B0': 1.0500, 'C0': 1.3000, 'D0': 0.001200, 'E0': 4.4800}}


# minimum redshift
ZMIN = 0.0

# maximum redshift
ZMAX = 5.0

# maximum of k (Mpc^-1)
KMAX = 5.0

# minimum of k (Mpc^-1)
KMIN = 5E-4

# number of k
NWAVE = 40

# number of redshift on the grid
NZ = 20

# new number of k (interpolated)
NK_NEW = 1000

# new number of z (interpolated)
NZ_NEW = 100

# curvature
OMEGA_K = 0.

# pivot scale in $ Mpc^{-1}$
K_PIVOT = 0.05

# Big Bang Nucleosynthesis
BBN = '/home/harry/Desktop/class/bbn/sBBN.dat'

# arguments to pass to CLASS
CLASS_ARGS = {"output": "mPk", "P_k_max_1/Mpc": KMAX, "z_max_pk": ZMAX}

# -----------------------------------------------------------------------------
# Priors
# specs are according to the scipy.stats. See documentation:
# https://docs.scipy.org/doc/scipy/reference/stats.html

# For example, if we want uniform prior between 1.0 and 5.0, then
# it is specified by loc and loc + scale, where scale=4.0
# distribution = scipy.stats.uniform(1.0, 4.0)

PRIORS = {

    'omega_cdm': {'distribution': 'uniform', 'specs': [0.06, 0.34]},
    'omega_b': {'distribution': 'uniform', 'specs': [0.019, 0.007]},
    'ln10^{10}A_s': {'distribution': 'uniform', 'specs': [1.70, 3.30]},
    'n_s': {'distribution': 'uniform', 'specs': [0.70, 0.60]},
    'h': {'distribution': 'uniform', 'specs': [0.64, 0.18]}
}

PARAMS = ['omega_cdm', 'omega_b', 'ln10^{10}A_s', 'n_s', 'h']

# choose which parameters to marginalise over
if NEUTRINO:
    PRIORS['M_tot'] = {'distribution': 'uniform', 'specs': [0.01, 0.99]}

if BARYON_FEEDBACK and MODE == 'hmcode':
    CLASS_ARGS['non linear'] = MODE
    PRIORS['c_min'] = {'distribution': 'uniform', 'specs': [2.0, 3.13]}
    PARAMS += ['c_min']
