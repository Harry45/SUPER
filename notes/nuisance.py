'''
Author: Arrykrishna Mootoovaloo
Collaborators: Alan Heavens, Andrew Jaffe, Florent Leclercq
Email : a.mootoovaloo17@imperial.ac.uk
Affiliation : Imperial Centre for Inference and Cosmology
Status : Under Development
Description : Calculate the model (linear model) associated with the nuisance parameters
'''

import os
from typing import Tuple
import numpy as np

# our scripts
import settings as st


class systematics(object):
    '''
    Calculate the model associated with the nuisance parameters
    '''

    def __init__(self):
        '''
        Load the basic files to calculate the systematics
        '''

        f_sigma = '{:}zbins/sigma_int_n_eff_{:}zbins.dat'.format(self.nzbins, self.nzbins)
        fname = os.path.join(st.data_directory, f_sigma)

        # read the data
        tbdata = np.loadtxt(fname)

        # choose the right data point
        sigma_e1 = tbdata[:, 2]
        sigma_e2 = tbdata[:, 3]
        n_eff = tbdata[:, 4]

        # calculate sigma_e
        self.sigma_e = np.sqrt((sigma_e1**2 + sigma_e2**2) / 2.)

        # convert from 1 / sq. arcmin to 1 / sterad
        self.n_eff = n_eff / np.deg2rad(1. / 60.)**2

    def noise_model(self, params: dict = None) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Calculate the noise associated with the band powers

        :param: params (dict) - a dictionary consisting of the nuisance parameters

        :return: (ee_bp_noisy, bb_np_noisy) - Tuple consisting the noise associated with each band power type
        '''

        # record noise
        a_noise = np.zeros(self.nzbins)

        # choose to add noise (boolean setting)
        add_noise_power = np.zeros(self.nzbins, dtype=bool)

        for zbin in range(self.nzbins):

            param_name = 'a{:}'.format(zbin + 1)

            if param_name in st.nuisance:
                a_noise[zbin] = params['a' + str(zbin + 1)]
                add_noise_power[zbin] = True

        # empty arrays for recording noise
        theory_bb = np.zeros((self.nzcorrs, self.band_offset_bb), 'float64')
        theory_noise_ee = np.zeros((self.nzcorrs, self.band_offset_ee), 'float64')
        theory_noise_bb = np.zeros((self.nzcorrs, self.band_offset_bb), 'float64')

        index_corr = 0

        for zbin1 in range(self.nzbins):
            for zbin2 in range(zbin1 + 1):

                if zbin1 == zbin2:
                    a_noise_corr = a_noise[zbin1] * self.sigma_e[zbin1]**2 / self.n_eff[zbin1]
                else:
                    a_noise_corr = 0.

                d_l_noise = self.ell_norm * a_noise_corr

                # because resetting bias is set to False
                # otherwise there is a random noise vector associated with the BB band powers
                theory_bb[index_corr, :] = 0.

                if add_noise_power.all():
                    theory_noise_ee[index_corr, :] = self.get_theory(d_l_noise, index_corr, band_type_is_ee=True)
                    theory_noise_bb[index_corr, :] = self.get_theory(d_l_noise, index_corr, band_type_is_ee=False)

                index_corr += 1

        # the noise associate for each band power
        ee_bp_noisy = theory_noise_ee.flatten()
        bb_bp_noisy = theory_noise_bb.flatten()

        return theory_bb, ee_bp_noisy, bb_bp_noisy
