'''
Author: Arrykrishna Mootoovaloo
Collaborators: Alan Heavens, Andrew Jaffe, Florent Leclercq
Email : a.mootoovaloo17@imperial.ac.uk
Affiliation : Imperial Centre for Inference and Cosmology
Status : Under Development
Description : Setup for the redshift distributions
'''

import os
import numpy as np
import scipy.interpolate as itp

# our scripts
import settings as st


class nz_dist(object):

    def __init__(self, file_settings='settings'):

        self.kids_data = st.data_directory

    def query_redshift(self, index: int = None):
        '''
        Extract the height and redshift. If an index (between 0 and 999) is specified, that particular
        n(z) distribution will be read

        :param: index (int) : Optional index. If not specified, the mean n(z) distrbution will be returned

        :return: red_args (dict) : A dictionary consisting of the redshift (key is z), and the 3 tomographic
        bins (keys are h0, h1 and h2)
        '''

        z_method = st.photoz_method

        # an empty list to store all redshifts
        self.z_bins = []

        # record the name of the type of n(z) distribution we want to use
        for index_zbin in range(len(st.zbin_min)):
            z_bin = '{:.2f}z{:.2f}'.format(st.zbin_min[index_zbin], st.zbin_max[index_zbin])
            self.z_bins.append(z_bin)

        # number of z-bins
        nzbins = len(self.z_bins)

        # generate a dictionary to record important quantities
        red_args = {}

        # iterate over each redshift bin and extract redshifts and height
        for xbin in range(nzbins):
            z_bin = self.z_bins[xbin]

            if index is None:

                # we use the average of the n(z) distribution
                z_name = '{:}/n_z_avg_{:}.hist'

                # specify the path of the file
                fname = os.path.join(self.kids_data, z_name.format(z_method, z_bin))

            else:

                z_name = '{:}/bootstraps/{:}/n_z_avg_bootstrap{:}.hist'

                fname = os.path.join(self.kids_data, z_name.format(z_method, z_bin, index))

            zptemp, hist_pz = np.loadtxt(fname, usecols=[0, 1], unpack=True)

            z, h = self.process_redshift(zptemp, hist_pz)

            # any height less than zero is assigned zero
            h[h < 1E-10] = 0.0

            red_args['h' + str(xbin)] = h

        red_args['z'] = z

        return red_args

    def generate_z_samples(self, random=False):
        '''
        Extact n(z) distributions - two options. Either we draw  random n(z) dstribution or we gather all the heights

        if random:
            red_args (dict) :  A dictionary consisting of the redshift (key is z), and the 3 tomographic bins
            (keys are h0, h1 and h2)

        else:
            red_args (dict) :  A dictionary consisting of the redshift (key is z), and the 3 tomographic bins
            (keys are h0, h1 and h2) but h0, h1 and h2 are samples now of size 1000 x 73

        :param: random (bool) : If True, a random integer between 0 and 999 (see settings file) will be drawn,
        else all samples are used

        :return: red_args (dict) : dictionary with the quantities described above
        '''

        if random:

            random_index = np.random.randint(int(st.index_bootstrap_low), int(st.index_bootstrap_high) + 1)

            red_args = self.query_redshift(index=random_index)

        else:

            nzbins = len(st.zbin_min)

            all_heights = [[] for i in range(nzbins)]

            for index in range(st.index_bootstrap_high + 1):

                red_args = self.query_redshift(index=index)

                for x in range(nzbins):
                    all_heights[x].append(red_args['h' + str(x)])

            # store the redshift
            z = red_args['z']

            # all samples of the n(z) distribution
            all_heights = np.asarray(all_heights)

            # create a new dictionary to store the samples
            red_args = {}

            red_args['z'] = z

            for i in range(nzbins):
                red_args['h' + str(i)] = all_heights[i]

        return red_args

    def process_redshift(self, redshift, height):
        '''
        Function to process the redshifts, that is shift them to mid point and normalise them properly

        :param: redshift (np.array) : an array of the original redshift

        :param: height (np.array) : an array for the heights of the n(z) distribution

        :return: redshifts (np.array) : the redshift but shifted to mid points

        :return: pr_red_norm (np.array) : the normalised redshift distribution
        '''

        shift_to_midpoint = np.diff(redshift)[0] / 2.

        z_samples = np.concatenate((np.zeros(1), redshift + shift_to_midpoint))
        hist_samples = np.concatenate((np.zeros(1), height))

        # Spline
        spline_pz = itp.splrep(z_samples, hist_samples)

        redshifts = z_samples[0:]
        mask_min = redshifts >= z_samples.min()
        mask_max = redshifts <= z_samples.max()
        mask = mask_min & mask_max

        # points outside the z-range of the histograms are set to 0!
        pr_red = itp.splev(redshifts[mask], spline_pz)

        # Normalize selection functions
        delta_z = redshifts[1:] - redshifts[:-1]
        pr_red_norm = pr_red / np.sum(0.5 * (pr_red[1:] + pr_red[:-1]) * delta_z)

        return redshifts, pr_red_norm
