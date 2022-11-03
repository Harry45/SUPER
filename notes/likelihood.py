'''
Author: Arrykrishna Mootoovaloo
Collaborators: Alan Heavens, Andrew Jaffe, Florent Leclercq
Email : a.mootoovaloo17@imperial.ac.uk
Affiliation : Imperial Centre for Inference and Cosmology
Status : Under Development
Description : Calculate the likelihood for KiDS-450
'''

from typing import Tuple
import numpy as np
import scipy.interpolate as itp
from scipy.linalg import cholesky, solve_triangular
from classy import Class

# our Python scripts
import settings as st
import cosmology.configurations as co
import cosmology.nuisance as ns
import cosmology.cosmofuncs as cf
import cosmology.spectrumcalc as sp
import cosmology.moped as mp
import utils.common as uc


class sampling_dist(co.data, ns.systematics, sp.matterspectrum, mp.compression):
    '''
    Calculate the likelihood for the KiDS-450 data

    :param: emulator (bool) - if True, the emulator will be used (Default: False)

    :param: director_gp (str) - if emulator is used, the correct GPs' directory must be specified (Default: False)

    :param: double_sum (bool) - choose whether we want to use the double sum power spectrum (Default: False)

    :param: nz_mean (bool) - choose whether we want to use n(z) samples or just the mean (Default: True)
    '''

    def __init__(
            self,
            emulator: bool = False,
            directory_gp: str = 'semigps',
            double_sum: bool = False,
            nz_mean: bool = True, **kwargs):

        self.emulator = emulator

        self.directory_gp = directory_gp

        self.double_sum = double_sum

        self.nz_mean = nz_mean

        # set th module to calculate the 3D matter power spectrum
        sp.matterspectrum.__init__(self, emulator)
        if self.emulator:
            sp.matterspectrum.load_gps(self, directory_gp)

        # set up all the configurations
        co.data.__init__(self)
        co.data.configs(self)

        # module for the systematics
        ns.systematics.__init__(self)

        # store the band powers, covariance matrix and cholesky factor at the fiducial value of m-correction factor
        # band powers, covariance, cholesky factor
        self.like_data = self.bandpowers_and_cov()

        # module for MOPED data compression
        if st.moped:
            mp.compression.__init__(self, model=self.kids_model, **kwargs)

            try:
                print('Loading MOPED vectors and compressed data if they exist.\n')
                mp.compression.load_moped(self)
            except Exception:
                print('Could not load MOPED vectors and compressed data.\n')
                print('Now calculating and storing them.\n')
                mp.compression.B_and_y(self, self.like_data[0], self.like_data[1], save=True)

    def m_vec_cov(self, mzbin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Calculate the vector and covariance of the m-correction

        :param: mzbin (np.ndarray) - array of the m-correction factor per tomgraphic bin

        :return: (m_vec, m_cov) - arrays of the vector and covariance used for re-scaling the bandpowers and
        data covariance matrix
        '''
        index_corr = 0

        for z1 in range(self.nzbins):
            for z2 in range(z1 + 1):

                # calculate m-correction vector here:
                # this loop goes over bands per z-corr
                # m-correction is the same for all bands in one tomographic bin.

                m1 = mzbin[z1]
                m2 = mzbin[z2]

                val_m_corr_ee = (1. + m1) * (1. + m2) * np.ones(len(st.bands_EE_to_use))
                val_m_corr_bb = (1. + m1) * (1. + m2) * np.ones(len(st.bands_BB_to_use))

                if index_corr == 0:
                    m_corr_ee = val_m_corr_ee
                    m_corr_bb = val_m_corr_bb
                else:
                    m_corr_ee = np.concatenate((m_corr_ee, val_m_corr_ee))
                    m_corr_bb = np.concatenate((m_corr_bb, val_m_corr_bb))

                index_corr += 1

        m_corr = np.concatenate((m_corr_ee, m_corr_bb))

        # this is required for scaling of covariance matrix:
        m_corr_matrix = np.matrix(m_corr).T * np.matrix(m_corr)

        return m_corr, m_corr_matrix

    def calc_m_correction(self, fiducial: bool = False, m_corr: float = None) -> np.ndarray:
        '''
        Calculates the m-correction vector and matrix. When using MOPED compression, the m-correction is fixed to the
        fiducial value.

        :param: fiducial (bool)- whether we want to use the fiducial value from KiDS-450

        :param: m_corr (float) - the m parameter

        :return: mzbin (np.ndarray) - the scaled m-correction per tomographic bin
        '''

        if fiducial:

            # if "m_corr" is not specified in input parameter script we just apply the fiducial m-correction values
            # if these could not be loaded, this vector contains only zeros!
            mzbin = self.m_corr_fiducial_per_zbin

        else:

            delta_m_corr = m_corr - self.m_corr_fiducial_per_zbin[0]

            mzbin = np.zeros(self.nzbins)

            for zbin in range(self.nzbins):
                mzbin[zbin] = self.m_corr_fiducial_per_zbin[zbin] + delta_m_corr

        return mzbin

    def bandpowers_and_cov(self, nuisance: dict = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Calculate the data and the covariance matrix, given the m_correction

        We also store the cholesky factor so that we don't invert the covariance matrix
        in each likelihood computation

        :param: nuisance (dict) - a dictionary consisting of the nuisance parameters

        :return: bp_new (np.ndarray) - the scaled band powers

        :return: cov_new (np.ndarray) - the scaled covariance matrix

        :return: chol_factor (np.ndarray) - the Cholesky factor based on the new covariance matrix
        '''

        # use the fiducial value
        if nuisance is None:

            mzbin = self.calc_m_correction(fiducial=True)

        # marginalise over the m nuisance parameter
        else:
            mzbin = self.calc_m_correction(fiducial=False, m_corr=nuisance['mc'])

        m, c = self.m_vec_cov(mzbin)

        # element-wise division of the covariance matrix
        cov_new = self.covariance / c

        # choose which elements are going to be used
        cov_new = cov_new[np.ix_(self.indices_for_bands_to_use, self.indices_for_bands_to_use)]

        # element-wise division of the band powers
        bp_new = self.band_powers / m

        # choose which band powers are to be used
        bp_new = bp_new[self.indices_for_bands_to_use]

        # computes the cholesky factor
        chol_factor = cholesky(cov_new, lower=True)

        return bp_new, cov_new, chol_factor

    def get_theory(self, d_l: np.ndarray, index_corr: np.ndarray, band_type_is_ee=True) -> np.ndarray:
        '''
        Compression 1: Power spectra are converted into band powers

        :param: d_l (np.ndarray): these are the interpolated power spectra

        :param: index_corr (int): the position in the auto- and cross-power, for example, 0: 00, 1: 01

        :param: band_type_is_ee (bool): True if we are working with ee band powers

        :return: d_avg (np.ndarray) - the band powers
        '''

        # these slice out the full ee --> ee and BB --> BB block of the full BWM!
        # sp: slicing points

        sp_ee_x = (0, self.nzcorrs * self.band_offset_ee)
        sp_ee_y = (0, self.nzcorrs * len(self.ells_intp))
        sp_bb_x = (self.nzcorrs * self.band_offset_ee, self.nzcorrs * (self.band_offset_bb + self.band_offset_ee))
        sp_bb_y = (self.nzcorrs * len(self.ells_intp), 2 * self.nzcorrs * len(self.ells_intp))

        if band_type_is_ee:
            sp_x = sp_ee_x
            sp_y = sp_ee_y
            band_offset = self.band_offset_ee
        else:
            sp_x = sp_bb_x
            sp_y = sp_bb_y
            band_offset = self.band_offset_bb

        bwm_sliced = self.bwm[sp_x[0]:sp_x[1], sp_y[0]:sp_y[1]]

        bands = range(index_corr * band_offset, (index_corr + 1) * band_offset)

        d_avg = np.zeros(len(bands))

        for index_band, alpha in enumerate(bands):
            # jump along tomographic auto-correlations only:
            il_low = int(index_corr * len(self.ells_intp))
            il_high = int((index_corr + 1) * len(self.ells_intp))
            spline_w_alpha_l = itp.splrep(self.ells_intp, bwm_sliced[alpha, il_low:il_high])
            d_avg[index_band] = np.sum(itp.splev(self.ells_sum, spline_w_alpha_l) * d_l)

        return d_avg

    def pk_matter(self, cosmo: dict, a_bary: float = 0.0) -> Tuple[dict, dict]:
        '''
        Calculate the non-linear matter power spectrum

        :param: d (dict) - a dictionary with all the parameters (keys and values)
        '''
        # get the cosmology part (input to simulator or emulator)

        # extract redshift (add a small jitter term for stability)
        redshifts = self.kids_z + 1E-300

        # calculate the basic quantities
        quant = self.basic_class(cosmo)

        # comoving radial distance
        chi = quant['chi']

        # we use the values of k and z where the GPs are built to build the interpolator
        k = self.k_range
        z = self.redshifts

        # get the predicted quantities from emulator or simulator
        if st.components:
            gf, spectrum, pl = sp.matterspectrum.int_pk_nl(self, params=cosmo, a_bary=a_bary, k=k, z=z)

        else:
            spectrum = sp.matterspectrum.int_pk_nl(self, params=cosmo, a_bary=a_bary, k=k, z=z)

        # emulator is trained with k in units of h Mpc^-1
        # therefore, we should input k = k/h in interpolator
        # example: interp(*[np.log(0.002/d['h']), 2.0])
        inputs = [k, z, spectrum.flatten()]

        interp = uc.like_interp_2d(inputs)

        # Get power spectrum P(k=l/r,z(r)) from cosmological module or emulator
        pk_matter = np.zeros((st.nellsmax, chi.shape[0]), 'float64')
        k_max_in_inv_mpc = st.kmax * cosmo['h']

        for il in range(st.nellsmax):
            for iz in range(1, chi.shape[0]):

                k_in_inv_mpc = (self.ells[il] + 0.5) / chi[iz]

                if k_in_inv_mpc > k_max_in_inv_mpc:

                    # assign a very small value of matter power
                    pk_dm = 1E-300

                else:

                    # the interpolator is built on top of log(k[h/Mpc])
                    newpoint = [np.log(k_in_inv_mpc / cosmo['h']), redshifts[iz]]

                    # predict the power spectrum
                    pk_dm = interp(*newpoint)

                # record the matter power spectrum
                pk_matter[il, iz] = pk_dm

        # record A_factor in the quant dictionary
        quant['a_fact'] = (3. / 2.) * quant['omega_m'] * quant['small_h']**2 / 2997.92458**2

        return pk_matter, quant

    def f_ell(self, cosmo: dict, a_bary: float = 0.0) -> Tuple[dict, dict]:
        '''
        Calcualte F_ell (chi) (see notes for further details) for the lensing power spectra (when we derive
        it as a double sum in terms of the heights of the n(z) distribution) and we calculate it for three
        types of power spectra: EE, GI, II

        :param: d (dict) : a dictionary for the input parameters (cosmology and nuisance)

        :return: f_ell (dict) : a dictionary for the F_ell(chi) computations

        :return: quant (dict) : a dictionary for the important quantities from CLASS
        '''

        # get the redshift
        redshifts = self.kids_z

        # calculate the non-linear matter power spectrum and important quantities
        pk_matter, quant = self.pk_matter(cosmo, a_bary)

        # comoving radial distance
        chi = quant['chi']

        # factor or the intrinsic alignment (of size 73)
        factor_ia = cf.get_factor_ia(quant, redshifts, 1.0)

        # calculate F_ell_chi (39 x 73) for EE power spectrum
        f_ell_ee = quant['a_fact']**2 * (1. + redshifts)**2 * pk_matter

        # calculate F_ell_chi (39 x 73) for the II power spectrum
        f_ell_ii = pk_matter * factor_ia**2 / chi**2

        # calculate F_ell_chi (39 x 73) for the GI power spectrum
        f_ell_gi = quant['a_fact'] * pk_matter * factor_ia * (1. + redshifts) / chi

        # a dictionary for all the f_ell calculated
        f_ell = {'ee': f_ell_ee, 'ii': f_ell_ii, 'gi': f_ell_gi}

        return f_ell, quant

    def ds_ps_calc(self, cosmo: dict, a_bary: float = 0.0) -> Tuple[dict, dict, dict]:
        '''
        Double sum power spectrum calculation

        Calculate the different weak lensing power spectra given a parameter

        :param: d (dict) - a dictionary consisting of the parameter names and values

        :return: cl (np.ndarray) - a dictionary for the different power spectra
        '''

        # calculate F_ell(chi)
        f, quant = self.f_ell(cosmo, a_bary)

        if self.nz_mean:
            zh = self.zh

        else:
            zh = self.generate_z_samples(random=True)

        # get the comoving radial distance
        chi = quant['chi']

        # calculate the different Q_ell
        q_ee_0 = cf.integration_q_ell(f['ee'], chi, 0)
        q_ee_1 = cf.integration_q_ell(f['ee'], chi, 1)
        q_ee_2 = cf.integration_q_ell(f['ee'], chi, 2)

        qs_ee = [q_ee_0, q_ee_1, q_ee_2]

        # calculate the double sum for each power spectrum type
        dsum_ee = cf.ds_ee(qs_ee, quant)
        dsum_ii = cf.ds_ii(f['ii'], quant)
        dsum_gi = cf.ds_gi(f['gi'], quant)

        # create emty dictionaries to store all power spectra
        cl_ee = {}
        cl_gi = {}
        cl_ii = {}

        for i in range(self.nzbins):
            for j in range(i + 1):

                idx = str(i) + str(j)

                # generate the power spectra (of length nk for each tomographic bin)
                ps_ee = cf.ps_ee(i, j, zh, dsum_ee)
                ps_ii = cf.ps_ii(i, j, zh, dsum_ii)
                ps_gi = cf.ps_gi(i, j, zh, dsum_gi)

                # interpolate the power spectra
                cl_ee[idx] = self.ell_norm * uc.interpolate([self.ells, ps_ee, self.ells_sum])
                cl_gi[idx] = self.ell_norm * uc.interpolate([self.ells, ps_gi, self.ells_sum])
                cl_ii[idx] = self.ell_norm * uc.interpolate([self.ells, ps_ii, self.ells_sum])

        return cl_ee, cl_gi, cl_ii

    def ff_ps_calc(self, cosmo: dict, a_bary: float = 0.0) -> Tuple[dict, dict, dict]:
        '''
        Power spectrum calculation using the functional form of the n(z) distribution

        :param: d (dict) - a dictionary for the parameters
        '''

        # get matter power spectrum and important quantities
        pk_matter, quant = self.pk_matter(cosmo, a_bary)

        # get the comoing radial distance
        chi = quant['chi']

        # A factor
        a_fact = quant['a_fact']

        # get the n(z) distributions
        if self.nz_mean:
            zh = self.zh

        else:
            zh = self.generate_z_samples(random=True)

        # n(z) to n(chi)
        pr_chi = np.array([zh['h' + str(i)] * quant['dzdr'] for i in range(self.nzbins)]).T

        kernel = np.zeros((st.nzmax + 1, self.nzbins), 'float64')

        for zbin in range(self.nzbins):
            for iz in range(1, st.nzmax + 1):
                fun = pr_chi[iz:, zbin] * (chi[iz:] - chi[iz]) / chi[iz:]
                kernel[iz, zbin] = np.sum(0.5 * (fun[1:] + fun[:-1]) * (chi[iz + 1:] - chi[iz:-1]))
                kernel[iz, zbin] *= chi[iz] * (1. + self.kids_z[iz])

        # Start loop over l for computation of C_l^shear
        cl_gg_int = np.zeros((st.nzmax + 1, self.nzbins, self.nzbins), 'float64')
        cl_ii_int = np.zeros_like(cl_gg_int)
        cl_gi_int = np.zeros_like(cl_gg_int)

        ps_ee = np.zeros((st.nellsmax, self.nzbins, self.nzbins), 'float64')
        ps_ii = np.zeros_like(ps_ee)
        ps_gi = np.zeros_like(ps_ee)

        # difference in chi (delta chi)
        dchi = chi[1:] - chi[:-1]

        # il refers to index ell
        for il in range(st.nellsmax):

            # find cl_int = (g(r) / r)**2 * P(l/r,z(r))
            for z1 in range(self.nzbins):
                for z2 in range(z1 + 1):

                    factor_ia = cf.get_factor_ia(quant, self.kids_z, 1.0)[1:]
                    fact_ii = pr_chi[1:, z1] * pr_chi[1:, z2] * factor_ia**2 / chi[1:]**2
                    fact_gi = kernel[1:, z1] * pr_chi[1:, z2] + kernel[1:, z2] * pr_chi[1:, z1]
                    fact_gi *= factor_ia / chi[1:]**2

                    cl_gg_int[1:, z1, z2] = kernel[1:, z1] * kernel[1:, z2] / chi[1:]**2 * pk_matter[il, 1:]
                    cl_ii_int[1:, z1, z2] = fact_ii * pk_matter[il, 1:]
                    cl_gi_int[1:, z1, z2] = fact_gi * pk_matter[il, 1:]

            for z1 in range(self.nzbins):
                for z2 in range(z1 + 1):
                    ps_ee[il, z1, z2] = np.sum(0.5 * (cl_gg_int[1:, z1, z2] + cl_gg_int[:-1, z1, z2]) * dchi)
                    ps_ii[il, z1, z2] = np.sum(0.5 * (cl_ii_int[1:, z1, z2] + cl_ii_int[:-1, z1, z2]) * dchi)
                    ps_gi[il, z1, z2] = np.sum(0.5 * (cl_gi_int[1:, z1, z2] + cl_gi_int[:-1, z1, z2]) * dchi)

                    ps_ee[il, z1, z2] *= a_fact**2
                    ps_gi[il, z1, z2] *= a_fact

        cl_ee = {}
        cl_gi = {}
        cl_ii = {}

        for z1 in range(self.nzbins):
            for z2 in range(z1 + 1):
                idx = str(z1) + str(z2)
                cl_ee[idx] = self.ell_norm * uc.interpolate([self.ells, ps_ee[:, z1, z2], self.ells_sum])
                cl_gi[idx] = self.ell_norm * uc.interpolate([self.ells, ps_gi[:, z1, z2], self.ells_sum])
                cl_ii[idx] = self.ell_norm * uc.interpolate([self.ells, ps_ii[:, z1, z2], self.ells_sum])

        return cl_ee, cl_gi, cl_ii

    def ps_to_bp(self, cosmo: dict, a_bary: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Compress all power spectra into a set of band powers

        :param: cosmo (dict) - a dictionary for the cosmological parameters

        :param: a_bary (float) - parameter for the baryon feedback

        :return: (bp_ee, bp_ii, bp_gi) - a tuple of the different band powers EE, II, GI
        '''

        # calculate the different power spectra
        if self.double_sum:
            cl_ee, cl_gi, cl_ii = self.ds_ps_calc(cosmo, a_bary)
        else:
            cl_ee, cl_gi, cl_ii = self.ff_ps_calc(cosmo, a_bary)

        bp_ee = np.zeros((self.nzcorrs, self.band_offset_ee), 'float64')
        bp_gi = np.zeros((self.nzcorrs, self.band_offset_ee), 'float64')
        bp_ii = np.zeros((self.nzcorrs, self.band_offset_ee), 'float64')

        index_corr = 0

        for z1 in range(self.nzbins):
            for z2 in range(z1 + 1):
                idx = str(z1) + str(z2)

                bp_ee[index_corr, :] = self.get_theory(cl_ee[idx], index_corr, band_type_is_ee=True)
                bp_gi[index_corr, :] = self.get_theory(cl_gi[idx], index_corr, band_type_is_ee=True)
                bp_ii[index_corr, :] = self.get_theory(cl_ii[idx], index_corr, band_type_is_ee=True)

                index_corr += 1

        return bp_ee, bp_gi, bp_ii

    def loglikelihood(self, parameters: dict) -> float:
        '''
        Calculate the log-likelihood using either band powers or MOPED

        :param: parameters (dict) - a dictionary of the parameters

        :return: loglike (float) - the log-likelihood value
        '''

        if st.moped:
            loglike = mp.compression.moped_loglike(self, parameters)

        else:
            loglike = self.bp_loglikelihood(parameters)

        return loglike

    def bp_loglikelihood(self, parameters: dict) -> float:
        '''
        Calculates log-likelihood of the data (band powers)

        :param: parameters (dict) - a dictionary of the parameters

        :return: loglike (float) - the log-likelihood value
        '''

        # the model
        bp_theory = self.kids_model(parameters)

        # get the data (depends on m-correction)
        if 'mc' in parameters:
            bp_data, cov_data, chol_data = self.bandpowers_and_cov(parameters)

        else:
            bp_data, cov_data, chol_data = self.like_data

        # calculate the differene between the data and the theory
        diff = bp_theory - bp_data

        # use triangular method to solve the equation
        y = solve_triangular(chol_data, diff, lower=True)

        # the determinant will change if m-correction is applied
        loglike = -0.5 * np.dot(y, y) - np.sum(np.diag(chol_data))

        return loglike

    def kids_model(self, parameters: dict) -> np.ndarray:
        '''
        Calculates the theretical band powers given a set of parameters

        :param: parameters (dict) - a dictionary of the parameters

        :return: bp_final (np.ndarry) - an array of the theoretical band powers
        '''

        # get the different dictionaries (cosmo, nuisance)
        # cosmology
        cosmology = cf.cosmo_params(parameters)

        # nuisance
        nuisance = cf.nuisance_params(parameters)

        # calculate the theoretical band powers
        bp_ee, bp_gi, bp_ii = self.ps_to_bp(cosmology, nuisance['a_bary'])

        # calculate total band powers
        bp_tot = bp_ee + nuisance['a_ia']**2 * bp_ii - nuisance['a_ia'] * bp_gi

        # calculate the term associated with the systematics part
        bp_bb, ee_noise, bb_noise = ns.systematics.noise_model(self, nuisance)

        # flatten all arrays
        bp_tot = bp_tot.flatten()
        bp_bb = bp_bb.flatten()
        ee_noise = ee_noise.flatten()
        bb_noise = bb_noise.flatten()

        # generate vectors of the theoretical part
        bp_final = np.concatenate((bp_tot, bp_bb)) + np.concatenate((ee_noise, bb_noise))

        # use only the selected band powers
        bp_final = bp_final[self.indices_for_bands_to_use]

        return bp_final

    def basic_class(self, cosmology: dict) -> dict:
        '''
        Calculates basic quantities using CLASS

        :param: d (dict) - a dictionary containing the cosmological and nuisance parameters

        :return: quant (dict) - a dictionary with the basic quantities
        '''

        cosmo, other, neutrino = cf.dictionary_params(cosmology)

        module = Class()

        # input cosmologies
        module.set(cosmo)

        # other settings for neutrino
        module.set(other)

        # neutrino settings
        module.set(neutrino)

        # compute basic quantities
        module.compute()

        # Omega_matter
        omega_m = module.Omega_m()

        # h parameter
        small_h = module.h()

        # critical density
        rc = cf.get_critical_density(small_h)

        # derive the linear growth factor D(z)
        lgr = np.zeros_like(self.kids_z)

        for iz, red in enumerate(self.kids_z):

            # compute linear growth rate
            lgr[iz] = module.scale_independent_growth_factor(red)

            # normalise linear growth rate at redshift = 0
            lgr /= module.scale_independent_growth_factor(0.)

        # get distances from cosmo-module
        chi, dzdr = module.z_of_r(self.kids_z)

        # numerical stability for chi
        chi += 1E-10

        # delete CLASS module to prevent memory overflow
        cf.delete_module(module)

        quant = {'omega_m': omega_m, 'small_h': small_h, 'chi': chi, 'dzdr': dzdr, 'lgr': lgr, 'rc': rc}

        # record the redshift as well

        quant['z'] = self.kids_z

        return quant
