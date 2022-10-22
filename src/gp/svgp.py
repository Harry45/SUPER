"""
Project: Scalable Gaussian Process Emulator (SUPER) for modelling power spectra
Author: Dr. Arrykrishna Mootoovaloo
Date: September 2022
Email: arrykrish@gmail.com

Implementation of Stochastic Variational Inference for Scalable
Gaussian Process. Example adapted from GPyTorch.
"""
from typing import Type, Union
import torch
import gpytorch

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from torch.utils.data import TensorDataset, DataLoader

# our scipt and functions
from .transformation import PreWhiten, Normalisation


class GPModel(ApproximateGP):
    """Creates a class for the approximate Gaussian Process model.

    Args:
        inducing_points (torch.Tensor): the set of inducing points.
        ndim (int): the number of dimensions for the RBF kernel
    """

    def __init__(self, inducing_points: torch.Tensor, ndim: int):

        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True)

        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ndim))

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Calculates the mean and covariance (kernel) of the training points.

        Args:
            xpoint (torch.Tensor): the training points.

        Returns:
            gpytorch.distributions.MultivariateNormal: a multivariate normal
            distribution with the calculated mean and covariance
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def dataloader(xtrain: torch.Tensor, ytrain: torch.Tensor, batch_size: int) -> DataLoader:
    """Creates a data loader for the data set.

    Args:
        batch_size (int): the batch size to use to split the full data set.
        xtrain (torch.Tensor): the input training points
        ytrain (torch.Tensor): the targets

    Returns:
        DataLoader: a data loader
    """
    train_dataset = TensorDataset(xtrain, ytrain)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader


def svgp_predictions(model: Type[GPModel], testpoint: torch.Tensor, var: bool) -> Union[torch.Tensor, torch.Tensor]:
    """Make predictions at test points in the parameter space.

    Args:
        model (Type[GPModel]): the trained GP model
        testpoint (torch.Tensor): the test point(s)
        var (bool): returns the variance if True

    Returns:
        Union[torch.Tensor, torch.Tensor]: mean and variance (optional)
    """
    preds = model(testpoint)
    return preds


class SVGaussianProcess(PreWhiten, Normalisation):
    """Builds a stochastic variational Gaussian Process given the training
    set and a set of inducing points.

    Args:
        xsamples (torch.Tensor): the inputs (x) of size N x d
        ysamples (torch.Tensor): the targets (y) of size N
        inducing (torch.Tensor): the set of inducing points of size M x d
    """

    def __init__(self, xsamples: torch.Tensor, ysamples: torch.Tensor, inducing: torch.Tensor):

        self.ndata = xsamples.shape[0]
        self.nind = inducing.shape[0]
        self.ndim = xsamples.shape[1]
        self.xsamples = xsamples
        self.ysamples = ysamples
        self.inducing = inducing
        self.trainedmodel: GPModel = None

        # apply transformation to inputs if the dimensionality is greater than 2
        if self.ndim >= 2:
            PreWhiten.__init__(self, xsamples)
            self.xtrain = PreWhiten.x_transformation(self, self.xsamples)
            self.ind_points = PreWhiten.x_transformation(self, self.inducing)
        else:
            self.xtrain = self.xsamples

        # apply normalisation to outputs
        Normalisation.__init__(self, ysamples)
        self.ytrain = Normalisation.y_transformation(self, self.ysamples)

    def training(self, lrate: float, nepochs: int, batch_size: int) -> list:
        """Train the model via the Adam optimization scheme.

        Args:
            lrate (float): the learning rate.
            nepochs (int): the number of epochs.
            batch_size (int): the batch size to use.

        Returns:
            list: a list of the recorded losses
        """

        model = GPModel(self.ind_points, self.ndim)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model.train()
        likelihood.train()
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=self.ndata)

        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()},
        ], lr=lrate)

        dataset = dataloader(self.xtrain, self.ytrain, batch_size)
        record_loss = list()
        for _ in range(nepochs):
            loss_per_epoch = 0.0
            for x_batch, y_batch in dataset:
                optimizer.zero_grad()
                output = model(x_batch)
                loss = -mll(output, y_batch)
                loss_per_epoch += loss.item()
                loss.backward()
                optimizer.step()
            record_loss.append(loss_per_epoch)
        self.trainedmodel = model
        return record_loss

    def predictions(self, testpoint: torch.Tensor, variance: bool) -> Union[torch.Tensor, torch.Tensor]:
        """Calculates the predictions given a test point or a batch of
        test points.

        Args:
            testpoint (torch.Tensor): one test point or a batch of test points
            variance (bool): if True, the variance will also be returned

        Returns:
            Union[torch.Tensor, torch.Tensor]: mean and variance (optional)
        """
        if self.ndim >= 2:
            test_trans = PreWhiten.x_transformation(self, testpoint)

        else:
            test_trans = testpoint
        multivariate = svgp_predictions(self.trainedmodel, test_trans, variance)
        mean = multivariate.mean.data
        ypred = Normalisation.y_inv_transformation(self, mean)
        if variance:
            yvar = self.ystd**2 * multivariate.variance.data
            return ypred, yvar
        return ypred
