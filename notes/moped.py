'''
Author: Arrykrishna Mootoovaloo
Collaborators: Alan Heavens, Andrew Jaffe, Florent Leclercq
Email : a.mootoovaloo17@imperial.ac.uk
Affiliation : Imperial Centre for Inference and Cosmology
Status : Under Development
Description : Performs MOPED data compression
'''

from typing import Tuple
import numpy as np
import utils.common as uc
import utils.helpers as hp
import cosmology.cosmofuncs as cf
import settings as st


class compression(object):
    '''
    Implementation of the MOPED algorithm

    :param: eps (list): step size value for computing finite differences

    :param: parameters (dict) - a dictionary of the parameters

    :param: model (object) - module for the forward model
    '''

    def __init__(self, eps: list, parameters: dict, model: object):

        eps = np.ones(1) * eps

        self.ndim = len(parameters)

        self.moped_param = uc.dvalues(parameters)

        if len(eps) == 1:
            eps = np.array([eps] * self.ndim)

        self.eps = eps

        self.model = model

    def B_and_y(self, data: np.array, cov: np.ndarray, save: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Compressing the data via MOPED (central finite difference method)

        :param: data (np.ndarray): the original data vector

        :param: cov (np.ndarray): the data covariance matrix

        :param: save (bool) : choose whether we want to save the MOPED vectors

        :return: bmat (np.ndarray): B matrix of size ndim x ndata

        :return: y_alphas (np.ndarray): the compressed data
        '''

        # compute inverse of the covariance matrix
        cov_inv = np.linalg.inv(cov)

        # number of data points
        n_data = len(data)

        # matrix to record the gradients
        grad = np.zeros((self.ndim, n_data))

        # some arrays to store important quantities
        cinv_grad_mu = np.zeros((self.ndim, n_data))
        grad_cinv_grad = np.zeros(self.ndim)
        bmat = np.zeros((self.ndim, n_data))

        # implementation of central finite difference method
        for i in range(self.ndim):

            # make a copy of the parameters
            p_plus = np.copy(self.moped_param)

            p_minus = np.copy(self.moped_param)

            # add and minus the step sizes
            p_plus[i] = p_plus[i] + self.eps[i]
            p_minus[i] = p_minus[i] - self.eps[i]

            # inputs to the model should be dictionaries
            p_plus = cf.mk_dict(st.marg_names, p_plus)
            p_minus = cf.mk_dict(st.marg_names, p_minus)

            # calculate the forward model at these points
            model_plus = self.model(p_plus)
            model_minus = self.model(p_minus)

            # calculate the gradients
            grad[i] = (model_plus - model_minus) / (2.0 * self.eps[i])

            # start of MOPED algorithm
            cinv_grad_mu[i] = np.dot(cov_inv, grad[i])
            grad_cinv_grad[i] = np.dot(grad[i], cinv_grad_mu[i])

        for i in range(self.ndim):

            if i == 0:
                bmat[i] = cinv_grad_mu[i] / np.sqrt(grad_cinv_grad[i])

            else:

                # numerator in the MOPED algorithm
                nume = np.zeros((n_data, int(i)))

                # denominator in the MOPED algorithm
                denom = np.zeros(int(i))

                for j in range(i):

                    # calculate the numerator
                    nume[:, j] = np.dot(grad[i], bmat[j]) * bmat[j]

                    # calculate the denominator
                    denom[j] = np.dot(grad[i], bmat[j])**2

                bmat[i] = (cinv_grad_mu[i] - np.sum(nume, axis=1)) / np.sqrt(grad_cinv_grad[i] - np.sum(denom))

        for i in range(self.ndim):
            for j in range(i + 1):
                if i == j:
                    msg = 'Dot product between {0:2d} and {1:2d} is : {2:.4f}'
                    ope = float(np.dot(bmat[i], np.dot(cov, bmat[j]).T))
                    print(msg.format(i, j, ope))

        # compute compressed data
        y_alphas = np.dot(bmat, data)

        # we also store the B matrix and the compressed data
        self.bmat = bmat
        self.y_alphas = y_alphas

        if save:
            if st.neutrino:
                hp.store_arrays(self.bmat, 'moped', 'bmat_neutrino')
                hp.store_arrays(self.y_alphas, 'moped', 'y_neutrino')

            else:
                hp.store_arrays(self.bmat, 'moped', 'bmat')
                hp.store_arrays(self.y_alphas, 'moped', 'y')

        return bmat, y_alphas

    def load_moped(self):
        '''
        Load the different moped vectors
        '''
        if st.neutrino:
            self.bmat = hp.load_arrays('moped', 'bmat_neutrino')
            self.y_alphas = hp.load_arrays('moped', 'y_neutrino')

        else:
            self.bmat = hp.load_arrays('moped', 'bmat')
            self.y_alphas = hp.load_arrays('moped', 'y')

    def expectation(self, parameters: dict) -> np.ndarray:
        '''
        Calculate the expectation value of the compressed data

        :param: parameters (dict) - a dictionary containing the parameter values

        :return: y_theory (np.ndarray) - an array of the expected data
        '''

        # calculate the expected theory
        theory = self.model(parameters)

        # compressed the data
        y_theory = np.dot(self.bmat, theory)

        return y_theory

    def moped_loglike(self, parameters: dict) -> float:
        '''
        Calculate the log-likelihood value using MOPED

        :param: parameters (dict) - a dictionary containing the parameter values

        :return: loglike (float) - the log-likelihood value
        '''

        # calculate the theory (MOPED) first
        y_theory = self.expectation(parameters)

        # difference between the compressed data and expected data
        diff = self.y_alphas - y_theory

        # calculate the log-likelihood
        loglike = -0.5 * np.sum(diff**2)

        return loglike
