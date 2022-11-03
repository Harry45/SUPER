'''
Author: Arrykrishna Mootoovaloo
Collaborators: Alan Heavens, Andrew Jaffe, Florent Leclercq
Email : a.mootoovaloo17@imperial.ac.uk
Affiliation : Imperial Centre for Inference and Cosmology
Status : Under Development
Description : class containing all the basic files (to be loaded for the likelihood computation)
'''

import os
import numpy as np

# our Python scripts
import cosmology.redshift as rd
import settings as st


class data(rd.nz_dist):

    def __init__(self):

        # module for the redshift
        rd.nz_dist.__init__(self)

    def configs(self):
        '''
        Load all important data files for the weak lensing analysis
        '''

        # create an empty list to record the redshift bins
        self.redshift_bins = []

        for i in range(len(st.zbin_min)):

            redshift_bin = '{:.2f}z{:.2f}'.format(st.zbin_min[i], st.zbin_max[i])

            self.redshift_bins.append(redshift_bin)

        # number of z-bins
        self.nzbins = len(self.redshift_bins)

        # number of *unique* correlations between z-bins
        self.nzcorrs = int(self.nzbins * (self.nzbins + 1) / 2)

        # which ee band powers to use (from setting file)
        self.all_bands_ee_to_use = []

        # which BB band powers to use (from setting file)
        self.all_bands_bb_to_use = []

        # default, use all correlations:
        for _ in range(self.nzcorrs):
            self.all_bands_ee_to_use += st.bands_EE_to_use
            self.all_bands_bb_to_use += st.bands_BB_to_use

        # all EE and BB bands to use
        self.all_bands_ee_to_use = np.array(self.all_bands_ee_to_use)
        self.all_bands_bb_to_use = np.array(self.all_bands_bb_to_use)

        # gather them in a single vector
        all_bands_to_use = np.concatenate((self.all_bands_ee_to_use, self.all_bands_bb_to_use))

        # find index where it is equal to 1 (that is the one to be used)
        self.indices_for_bands_to_use = np.where(np.asarray(all_bands_to_use) == 1)[0]

        # m-correction
        fname = os.path.join(st.data_directory, '{:}zbins/m_correction_avg.txt'.format(self.nzbins))

        self.m_corr_fiducial_per_zbin = np.loadtxt(fname, usecols=[1])

        # load the band window matrix
        fname = os.path.join(st.data_directory, '{:}zbins/band_window_matrix_nell100.dat'.format(self.nzbins))

        self.bwm = np.loadtxt(fname)

        # load the band powers (ee and BB)
        collect_bp_ee_in_zbins = []
        collect_bp_bb_in_zbins = []
        # collect BP per zbin and combine into one array
        for zbin1 in range(self.nzbins):
            for zbin2 in range(zbin1 + 1):

                f_ee = '{:}zbins/band_powers_EE_z{:}xz{:}.dat'.format(self.nzbins, zbin1 + 1, zbin2 + 1)
                f_bb = '{:}zbins/band_powers_BB_z{:}xz{:}.dat'.format(self.nzbins, zbin1 + 1, zbin2 + 1)

                fname_ee = os.path.join(st.data_directory, f_ee)
                fname_bb = os.path.join(st.data_directory, f_bb)

                extracted_band_powers_ee = np.loadtxt(fname_ee)
                extracted_band_powers_bb = np.loadtxt(fname_bb)

                collect_bp_ee_in_zbins.append(extracted_band_powers_ee)
                collect_bp_bb_in_zbins.append(extracted_band_powers_bb)

        bp_ee = np.asarray(collect_bp_ee_in_zbins).flatten()
        bp_bb = np.asarray(collect_bp_bb_in_zbins).flatten()
        # band powers
        self.band_powers = np.concatenate((bp_ee, bp_bb))

        # Load the covariance matrix
        fname = os.path.join(st.data_directory, '{:}zbins/covariance_all_z_EE_BB.dat'.format(self.nzbins))
        self.covariance = np.loadtxt(fname)

        # ells_intp and also band_offset are consistent between different patches!
        bwf = '{:}zbins/multipole_nodes_for_band_window_functions_nell100.dat'.format(self.nzbins)
        fname = os.path.join(st.data_directory, bwf)
        self.ells_intp = np.loadtxt(fname)

        self.band_offset_ee = len(extracted_band_powers_ee)
        self.band_offset_bb = len(extracted_band_powers_bb)

        # other important quantities required
        self.ells_min = self.ells_intp[0]
        self.ells_max = self.ells_intp[-1]
        self.nells = int(self.ells_max - self.ells_min + 1)

        # these are the \ell modes
        self.ells_sum = np.linspace(self.ells_min, self.ells_max, self.nells)

        # these are the l-nodes for the derivation of the theoretical cl:
        # \ells in logspace
        self.ells = np.logspace(np.log10(self.ells_min), np.log10(self.ells_max), st.nellsmax)

        # normalisation factor
        self.ell_norm = self.ells_sum * (self.ells_sum + 1) / (2. * np.pi)

        # bands selected (indices only)
        self.bands_ee_selected = np.tile(st.bands_EE_to_use, self.nzcorrs)
        self.bands_bb_selected = np.tile(st.bands_BB_to_use, self.nzcorrs)

        # redshift configurations (redshift and mean height)
        self.zh = rd.nz_dist.query_redshift(self)

        # we use redshift in many parts of the code
        self.kids_z = self.zh['z']

        # self.zh = rd.nz_dist.generate_z_samples(self, random=True)
