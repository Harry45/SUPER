"""
Project: Scalable Gaussian Process Emulator (SUPER) for modelling power spectra
Author: Dr. Arrykrishna Mootoovaloo
Date: September 2022
Email: arrykrish@gmail.com
Description: Sparse Gaussian Process usinf Inducing Points
"""

import torch
import torch.autograd
import src.gp.kernel as kn
from src.gp.transformation import PreWhiten, Normalisation


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
        self.sigma = sigma

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

        # first term (the trace term)
        qmq = q_matrix @ kn.solve(m_matrix, q_matrix.t())
        k_tilde = k_matrix - qmq
        term1 = torch.trace(k_tilde) / self.sigma**2

        # second term (data fit term)
        mat1 = m_matrix + self.beta * q_matrix.t() @ q_matrix
        mat2 = q_matrix @ kn.solve(mat1, q_matrix.t())
        mat3 = self.beta - self.beta**2 * mat2
        term2 = self.ytrain.t() @ mat3 @ self.ytrain

        # third term (model complexity term)
        determinant = torch.det(self.jitter**2 + qmq)
        print(determinant)
        assert determinant >= 0.0, 'The determinant term is negative'

        # calculate total
        total = -0.5 * term1 - 0.5 * term2 - 0.5 * torch.log(determinant)

        return -total
