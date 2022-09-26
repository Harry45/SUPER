"""
Project: Scalable Gaussian Process Emulator (SUPER) for modelling power spectra
Author: Dr. Arrykrishna Mootoovaloo
Date: September 2022
Email: arrykrish@gmail.com
Description: Sparse Gaussian Process usinf Inducing Points
"""

import torch
import torch.autograd
import numpy as np
import src.gp.kernel as kn
from src.gp.transformation import PreWhiten, Normalisation


def logdet_chol(chol_factor: torch.Tensor) -> torch.tensor:
    """Calculates the log-determinant of a matrix, given its Cholesky factor.

    Args:
        chol_factor (torch.Tensor): the Cholesky factor

    Returns:
        torch.tensor: the log-determinant value
    """

    return 2 * torch.sum(torch.log(torch.diag(chol_factor)))


def trace_term(kernel: torch.Tensor, q_matrix: torch.Tensor,
               m_chol_matrix: torch.Tensor, sigma: torch.tensor) -> torch.Tensor:
    """Calculates the trace term (additional regularisation term) in the
    calculation of the marginal likelihood

    Args:
        kernel (torch.Tensor): the kernel matrix of size n x n
        q_matrix (torch.Tensor): the Q matrix of size n x m
        m_chol_matrix (torch.Tensor): the Cholesky factor of matrix M
        sigma (torch.tensor): the noise term

    Returns:
        torch.Tensor: the value of the trace term
    """

    k_tilde = kernel - q_matrix @ torch.cholesky_solve(q_matrix.t(), m_chol_matrix)
    trace = sigma**-2 * torch.trace(k_tilde)
    return -0.5 * trace


def determinant_term(q_matrix: torch.Tensor, m_matrix: torch.Tensor,
                     m_chol_matrix: torch.Tensor, sigma: torch.tensor) -> torch.tensor:
    """Calculates the log determinant term using the Matrix determinant lemma.
    This is done so that we can work with the number of inducing points.

    Args:
        q_matrix (torch.Tensor): the Q matrix of size n x m
        m_matrix (torch.Tensor): the M matrix of size M x M
        m_chol_matrix (torch.Tensor): the Cholesky factor of matrix M
        sigma (torch.tensor): the noise term

    Returns:
        torch.tensor: the log determinant value
    """
    num_data = q_matrix.shape[0]
    sigma2_inv = sigma**-2 * torch.eye(num_data)
    p_matrix = q_matrix.t() @ torch.mm(sigma2_inv, q_matrix) + m_matrix

    p_matrix_chol = torch.linalg.cholesky(p_matrix)
    log_det = 2 * num_data * torch.log(sigma)
    log_det += logdet_chol(p_matrix_chol)
    log_det -= logdet_chol(m_chol_matrix)
    return -0.5 * log_det, p_matrix_chol


def data_fit_term(outputs: torch.Tensor, p_matrix_chol: torch.Tensor,
                  q_matrix: torch.Tensor, sigma: torch.tensor) -> torch.Tensor:
    """Calculates the data-fit term given the different matrices.

    Args:
        outputs (torch.Tensor): the observed values (y in the notes)
        p_matrix_chol (torch.Tensor): the Cholesky factor of the matrix:

        sigma**-2 Q^T Q + M

        q_matrix (torch.Tensor): the Q matrix of size n x m
        sigma (torch.tensor): the noise term

    Returns:
        torch.Tensor: the data fit term
    """
    num_data = q_matrix.shape[0]

    # reshape outputs
    outputs = outputs.view(-1, 1)

    # calculate Sigma^-1 Q
    sigma2_inv = sigma**-2 * torch.eye(num_data)
    sigma2_inv_q = sigma2_inv @ q_matrix

    # calculate the inverse
    term = torch.cholesky_solve(sigma2_inv_q.t(), p_matrix_chol)
    term = sigma2_inv_q @ term
    term = sigma2_inv - term

    data_fit = -0.5 * outputs.t() @ term @ outputs

    return data_fit[0, 0]


def calculate_alpha(outputs: torch.Tensor, q_matrix: torch.Tensor,
                    m_matrix: torch.Tensor, sigma: torch.tensor) -> torch.Tensor:
    """Calculates the weights (the alpha vector) given the different matrices

    Args:
        outputs (torch.Tensor): the observed data of size n
        q_matrix (torch.Tensor): the Q matrix of size n x m
        m_matrix (torch.Tensor): the M matrix of size m x m
        sigma (torch.tensor): the noise term

    Returns:
        torch.Tensor: the vector alpha of size m
    """
    p_matrix = sigma**-2 * q_matrix.t() @ q_matrix + m_matrix
    p_matrix_chol = torch.linalg.cholesky(p_matrix)
    m_matrix_chol = torch.linalg.cholesky(m_matrix)
    q_trans_y = q_matrix.t() @ outputs
    q_trans_y = q_trans_y.view(-1, 1)
    alpha = sigma**-2 * torch.cholesky_solve(q_trans_y, p_matrix_chol)
    return alpha, p_matrix_chol, m_matrix_chol


class GaussianProcess(PreWhiten, Normalisation):
    """Implementation of a zero mean Sparse Gaussian Process using inducing point method.

    Args:
        inputs (torch.Tensor): the inputs to the model, of size n x d
        targets (torch.Tensor): the targets (outputs), of size n x 1
        inducing (torch.Tensor): the inducing points, of size m x d
        jitter (float): the noise term (we are assuming uncorrelated
        Gaussian noise)
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor,
                 inducing: torch.Tensor, sigma: float):

        # store all relevant quantities
        self.inputs = inputs
        self.targets = targets
        self.inducing = inducing
        self.sigma = torch.tensor(sigma)

        # store all important quantities
        self.alpha = None
        self.p_matrix_chol = None
        self.m_matrix_chol = None
        self.d_opt = None
        self.opt_parameters = None

        # get all the relevant dimensions
        self.numind = self.inducing.shape[0]
        self.ndata, self.ndim = self.inputs.shape
        assert (self.ndata > self.ndim), 'not enough training points or reshape tensor'

        # apply transformation
        PreWhiten.__init__(self, self.inputs)
        Normalisation.__init__(self, self.targets)
        self.apply_transformation()

    def apply_transformation(self):
        """Apply transformation to the input and targets.
        """

        # apply transformation to inputs if the dimensionality is greater than 2
        if self.ndim >= 2:
            self.xtrain = PreWhiten.x_transformation(self, self.inputs)
            self.inducing_trans = PreWhiten.x_transformation(self, self.inducing)
        else:
            self.xtrain = self.inputs

        # apply normalisation to outputs
        self.ytrain = Normalisation.y_transformation(self, self.targets)

    def cost_function(self, parameter: torch.Tensor) -> torch.Tensor:
        """Calculates the negative log-likelihood for the Gaussian Process.

        Args:
            parameter (torch.Tensor): the kernel hyperparameters

        Returns:
            torch.Tensor: the value of the negatice log-likelihood
        """
        # covariance between training points (n x n)
        k_matrix = kn.compute(self.xtrain, self.xtrain, parameter)

        # covariance due to the inducing points (m x m)
        m_matrix = kn.compute(self.inducing_trans, self.inducing_trans, parameter)

        # cross covariance between training and inducing points (n x m)
        q_matrix = kn.compute(self.xtrain, self.inducing_trans, parameter)

        # calculates the Cholesky factor of matrix M
        m_chol_factor = torch.linalg.cholesky(m_matrix)

        # calculates the different parts
        trace = trace_term(k_matrix, q_matrix, m_chol_factor, self.sigma)
        det, p_matrix_chol = determinant_term(q_matrix, m_matrix, m_chol_factor, self.sigma)
        fit = data_fit_term(self.ytrain, p_matrix_chol, q_matrix, self.sigma)

        return -1.0 * (trace + det + fit)

    def optimisation(self, parameters: torch.Tensor, niter: int = 10, lrate: float = 0.01, nrestart: int = 2) -> dict:
        """Optimise for the kernel hyperparameters using Adam in PyTorch.
        Args:
            parameters(torch.tensor): a tensor of the kernel hyperparameters.
            niter(int): the number of iterations we want to use
            lr(float): the learning rate
            nrestart(int): the number of times we want to restart the optimisation
        Returns:
            dict: dictionary consisting of the optimised values of the hyperparameters and the loss.
        """

        dictionary = {}

        for i in range(nrestart):

            # make a copy of the original parameters and perturb it
            params = parameters.clone() + torch.randn(parameters.shape) * 0.1  # torch.randn(parameters.shape)  #

            # make sure we are differentiating with respect to the parameters
            params.requires_grad = True

            # initialise the optimiser
            optimiser = torch.optim.Adam([params], lr=lrate)

            loss = self.cost_function(params)

            # an empty list to store the loss
            record_loss = [loss.item()]

            # run the optimisation
            for _ in range(niter):

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                # evaluate the loss
                loss = self.cost_function(params)

                # record the loss at every step
                record_loss.append(loss.item())
                print(loss)

            dictionary[i] = {"parameters": params, "loss": record_loss}

        # get the dictionary for which the loss is the lowest
        self.d_opt = dictionary[np.argmin([dictionary[i]["loss"][-1] for i in range(nrestart)])]

        # store the optimised parameters as well
        self.opt_parameters = self.d_opt["parameters"]

        # to update the matrices and compute important quantities for making
        # predictions
        # covariance due to the inducing points (m x m)
        m_matrix = kn.compute(self.inducing_trans, self.inducing_trans, self.opt_parameters)

        # cross covariance between training and inducing points (n x m)
        q_matrix = kn.compute(self.xtrain, self.inducing_trans, self.opt_parameters)
        alpha, p_matrix_chol, m_matrix_chol = calculate_alpha(self.ytrain, q_matrix, m_matrix, self.sigma)

        # store all important quantities
        self.alpha = alpha
        self.p_matrix_chol = p_matrix_chol
        self.m_matrix_chol = m_matrix_chol

        return dictionary

    def prediction(self, testpoint: torch.Tensor, variance: bool = False):

        if self.ndim >= 2:
            point = PreWhiten.x_transformation(self, testpoint)
            print(point)
        else:
            point = testpoint

        k_star = kn.compute(self.inducing_trans, point, self.opt_parameters)
        pred = k_star.t() @ self.alpha
        print(pred)
        print(Normalisation.y_inv_transformation(self, pred))
