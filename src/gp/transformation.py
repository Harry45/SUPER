"""
Project: Scalable Gaussian Process Emulator (SUPER) for modelling power spectra
Author: Dr. Arrykrishna Mootoovaloo
Date: September 2022
Email: arrykrish@gmail.com
Description: Code to transform the inputs to the GP, that is, the input training points
"""

import torch


class Normalisation:
    """
    Normalise the values of y.

    Args:
        outputs (torch.Tensor): the values of y (in GP formalism)
    """

    def __init__(self, yinputs: torch.Tensor):
        self.yinputs = yinputs
        self.yinputs = self.yinputs.view(-1, 1)

        # transformation
        self.ymean = torch.mean(self.yinputs)
        self.ystd = torch.std(self.yinputs)

    def y_transformation(self, yvalue: torch.Tensor) -> torch.Tensor:
        """Transform the data such that the transformed data has a mean of 0 and
        a standard deviation of 1.

        Args:
            yvalue (torch.Tensor): the values of y

        Returns:
            torch.Tensor: the transformed values of y
        """
        return (yvalue - self.ymean) / self.ystd

    def y_inv_transformation(self, value: torch.Tensor) -> torch.Tensor:
        """The inverse transform on y.

        Args:
            value (torch.Tensor): the value of y (predicted from the GP)

        Returns:
            torch.Tensor: the predicted value of y (inverse transform)
        """
        return self.ystd * value + self.ymean


class PreWhiten:
    """
    Prewhiten the inputs such that the resulting mean of the inputs is zero
    and the covariance is the identity matrix.

    Args:
        xinputs (torch.tensor): the inputs of size n x d.
    """

    def __init__(self, xinputs: torch.Tensor):
        self.xinputs = xinputs

        assert self.xinputs.shape[0] > self.xinputs.shape[1], 'Reshape your tensors'

        self.ndim = self.xinputs.shape[1]

        # compute the covariance of the inputs (ndim x ndim)
        self.cov_train = torch.cov(self.xinputs.t())

        # compute the Cholesky decomposition of the matrix
        self.chol_train = torch.linalg.cholesky(self.cov_train)

        # compute the mean of the sample
        self.mean_train = torch.mean(self.xinputs, axis=0).view(1, self.ndim)

    def x_transformation(self, point: torch.tensor) -> torch.tensor:
        """Pre-whiten the input parameters.

        Args:
            point (torch.tensor): the input parameters.

        Returns:
            torch.tensor: the pre-whitened parameters.
        """

        # ensure the point has the right dimensions
        point = point.view(-1, self.ndim)

        # calculate the transformed training points
        transformed = torch.linalg.inv(self.chol_train) @ (point - self.mean_train).t()

        return transformed.t()
