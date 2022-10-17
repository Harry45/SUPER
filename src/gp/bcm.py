"""
Project: Scalable Gaussian Process Emulator (SUPER) for modelling power spectra
Author: Dr. Arrykrishna Mootoovaloo
Date: September 2022
Email: arrykrish@gmail.com
Description: Functions related to the Gaussian Process
"""

import torch
from typing import Tuple, Union
import numpy as np
from sklearn.cluster import KMeans

# our script and functions
from src.gp.kernel import solve, logdeterminant, compute_kernel
from src.gp.transformation import PreWhiten, Normalisation


def clustering(xsamples: torch.Tensor, ysamples: torch.Tensor, n_clusters: int) -> dict:
    """Apply clustering to the input data and generate the clustered batch data
    set.

    Args:
        xsamples (torch.Tensor): the input to the model
        ysamples (torch.Tensor): the target (the power spectrum in this case)
        n_clusters (int): the number of clusters

    Returns:
        dict: a dictionary to store the partitioned data
    """
    cluster_module = KMeans(n_clusters, random_state=0).fit(xsamples.data.numpy())
    record = {}
    for i in range(n_clusters):
        labels = cluster_module.labels_ == i
        xpoints = xsamples[labels]
        ypoints = ysamples[labels]
        record[str(i)] = (xpoints, ypoints)
    return cluster_module, record


def distance_from_cluster(module: KMeans, xtest: torch.Tensor) -> np.ndarray:
    """Calculates the paiwise distance between test point to each centre of the
    cluster.

    Args:
        module (KMeans): The KMeans module, already fitted on the training
        points
        xtest (torch.Tensor): The test point

    Returns:
        np.ndarray: an array of sorted index of each cluster
    """
    xtest = xtest.view(1, -1)
    pdist = torch.nn.PairwiseDistance(p=2)
    record = list()
    for i in range(module.n_clusters):
        centre = module.cluster_centers_[i].reshape(1, -1)
        centre = torch.from_numpy(centre)
        distance = pdist(xtest, centre)
        record.append(distance)

    sorted_index = np.argsort(record)
    return sorted_index


def log_marginal_likelihood(kernel: torch.Tensor, targets: torch.Tensor, jitter: float) -> torch.tensor:
    """Calculates the log marginal likelihood given a kernel and the targets (y values)

    Args:
        kernel (torch.Tensor): the kernel matrix of size m
        targets (torch.Tensor): the targets of size m
        jitter (float): a jitter term for numerical stability

    Returns:
        torch.tensor: the value of the log-likelihood
    """
    kernel = kernel + torch.eye(kernel.shape[0]) * jitter
    targets = targets.view(-1, 1)
    alpha = solve(kernel, targets)
    chi2 = targets.t() @ alpha
    logdet = logdeterminant(kernel)
    log_ml = -0.5 * (chi2 + logdet)
    return log_ml.view(-1)


def cost(record: dict, hyperparameters: torch.Tensor, jitter: float) -> torch.Tensor:
    """Calculates the cost function

    Args:
        record (dict): A dictionary consisting of the partitioned inputs and
        targets
        hyperparameters (torch.Tensor): the set of hyperparameters
        jitter (float): a jitter term for numerical stability

    Returns:
        torch.Tensor: the approximate total cost
    """
    nclusters = len(record)
    value = torch.zeros(1)
    for i in range(nclusters):
        xpoint, ypoint = record[str(i)][0], record[str(i)][1]
        kernel = compute_kernel(xpoint, xpoint, hyperparameters)
        log_ml = log_marginal_likelihood(kernel, ypoint, jitter)
        value = value + log_ml
    return value


def cost_exact(xpoint: torch.Tensor, ypoint: torch.Tensor,
               hyperparameters: torch.Tensor, jitter: float) -> torch.Tensor:
    """Calculates the cost function

    Args:
        xpoint (torch.Tensor):
        ypoint (torch.Tensor):
        hyperparameters (torch.Tensor): the set of hyperparameters
        jitter (float): a jitter term for numerical stability

    Returns:
        torch.Tensor: the log marginal likelihood
    """
    kernel = compute_kernel(xpoint, xpoint, hyperparameters)
    log_ml = log_marginal_likelihood(kernel, ypoint, jitter)
    return log_ml


def kernel_alpha(record: dict, parameters: torch.Tensor, jitter: float) -> Tuple[dict, dict]:
    """Calculates the final kernel and the weights (alpha).

    Args:
        record (dict): A dictionary containing the inputs and targets.
        parameters (torch.Tensor): The optimised parameters of the kernel
        jitter (float): The jitter term for numerical stability

    Returns:
        Tuple[dict, dict]: A dictionary of the kernel and a dictionary of alpha
    """

    nclusters = len(record)
    kernels = {}
    alphas = {}
    for i in range(nclusters):
        xpoint, ypoint = record[str(i)][0], record[str(i)][1]
        kernel = compute_kernel(xpoint, xpoint, parameters)
        kernel = kernel + torch.eye(kernel.shape[0]) * jitter
        alpha = solve(kernel, ypoint)
        kernels[str(i)] = kernel
        alphas[str(i)] = alpha

    return kernels, alphas


def poe_predictions(means: torch.Tensor, variances: torch.Tensor) -> dict:
    """Calculates the mean and variance predictions using the Product of Expert method.

    Args:
        means (torch.Tensor): A tensor with the mean calculated using each expert.
        variance (torch.Tensor): A tensor with the variance calculated using each expert.

    Returns:
        dict: A dictionary with the final mean and variance.
    """
    pred_var = 1. / torch.sum(1. / variances)
    pred_mean = pred_var * torch.sum(means / variances)
    predictions = {}
    predictions['mean'] = pred_mean
    predictions['variance'] = pred_var
    return predictions


def bcm_predictions(means: torch.Tensor, variances: torch.Tensor, prior_variance: torch.Tensor) -> dict:
    """Calculates the mean and variance predictions using Bayesian Committee Machine

    Args:
        means (torch.Tensor): A tensor of the mean calculated from each expert.
        variances (torch.Tensor): A tensor of the variance from each expert.
        prior_variance (torch.Tensor): The prior variance, f* ~ N(0, A)

    Returns:
        dict: A dictionary with the final mean and variance
    """
    n_neighbour = len(means)
    precision = (1. - n_neighbour) / prior_variance + torch.sum(1. / variances)
    pred_var = 1. / precision
    pred_mean = pred_var * torch.sum(means / variances)
    predictions = {}
    predictions['mean'] = pred_mean
    predictions['variance'] = pred_var
    return predictions


def approximate_predictions(xtest: torch.Tensor, inputs: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculates the approximate predictions using the experts - either PoE or BCM

    Args:
        xtest (torch.Tensor): the input test point
        inputs (dict): a dictionary with all the relevant quantities for calculations

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: mean and variance of the predicted function
    """


def exact_predictions(xtest: torch.Tensor, inputs: dict, var: bool) -> Union[torch.Tensor, torch.Tensor]:
    """Calculates the exact prediction, that is, we are using a full GP.

    Args:
        xtest (torch.Tensor): the input test point
        inputs (dict): a dictionary with all the relevant quantities for calculations
        var (bool): if we want to return the variance as well

    Returns:
        Union[torch.Tensor, torch.Tensor]: mean OR mean and variance
    """
    xtrain = inputs['xtrain']
    hyper = inputs['hyper']
    kernel = inputs['kernel']
    alpha = inputs['alpha']
    ytrain = inputs['ytrain']

    k_star = compute_kernel(xtest, xtrain, hyper)
    mean = k_star @ alpha
    if var:
        k_ss = compute_kernel(xtest, xtest, hyper)
        variance = k_ss - k_star @ solve(kernel, ytrain)
        return mean, variance
    return mean


class BayesianMachine(PreWhiten, Normalisation):

    def __init__(self, xsamples: torch.Tensor, ysamples: torch.Tensor, sigma: float, **kwargs):

        self.exact = True
        self.ndim = xsamples.shape[1]
        self.ndata = xsamples.shape[0]
        assert (self.ndata > self.ndim), 'not enough training points or reshape tensor'

        # store all the data
        self.xsamples = xsamples
        self.ysamples = ysamples.view(-1, 1)
        self.sigma = sigma

        # apply transformation
        PreWhiten.__init__(self, xsamples)
        Normalisation.__init__(self, ysamples)

        # apply transformation to inputs if the dimensionality is greater than 2
        if self.ndim >= 2:
            self.xtrain = PreWhiten.x_transformation(self, self.xsamples)
        else:
            self.xtrain = self.xsamples

        # apply normalisation to outputs
        self.ytrain = Normalisation.y_transformation(self, self.ysamples)

        if 'n_clusters' in kwargs:
            self.exact = False
            n_clusters = kwargs.pop('n_clusters')
            self.cluster_module, self.record = clustering(self.xtrain, self.ytrain, n_clusters)

    def optimisation(self, parameters: torch.Tensor, configurations: dict) -> dict:
        """Optimise for the kernel hyperparameters using Adam in PyTorch.
        Args:
            parameters(torch.tensor): a tensor of the kernel hyperparameters.
            configurations(dict): a dictionary with the following parameters:
                - niter(int): the number of iterations we want to use
                - lr(float): the learning rate
                - nrestart(int): the number of times we want to restart the optimisation
        Returns:
            dict: dictionary consisting of the optimised values of the hyperparameters and the loss.
        """
        niter = configurations['niter']
        nrestart = configurations['nrestart']
        lrate = configurations['lrate']

        dictionary = {}
        for i in range(nrestart):
            params = parameters.clone() + torch.randn(parameters.shape) * 0.1
            params.requires_grad = True
            optimiser = torch.optim.Adam([params], lr=lrate)

            if self.exact:
                loss = cost_exact(self.xtrain, self.ytrain, params, self.sigma)
            else:
                loss = cost(self.record, params, self.sigma)
            record_loss = [loss.item()]

            for _ in range(niter):
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                if self.exact:
                    loss = cost_exact(self.xtrain, self.ytrain, params, self.sigma)
                else:
                    loss = cost(self.record, params, self.sigma)
                record_loss.append(loss.item())
            dictionary[i] = {"parameters": params.data, "loss": record_loss}

        # store relevant quantities
        self.d_opt = dictionary[np.argmin([dictionary[i]["loss"][-1] for i in range(nrestart)])]
        self.opt_parameters = self.d_opt["parameters"]
        if self.exact:
            self.kernel = compute_kernel(self.xtrain, self.xtrain, self.opt_parameters)
            self.kernel = self.kernel + torch.eye(self.ndata) * self.sigma
            self.alpha = solve(self.kernel, self.ytrain)
        else:
            self.kernels, self.alphas = kernel_alpha(self.record, self.opt_parameters, self.sigma)

        return dictionary

    def mean_prediction(self, testpoint: torch.Tensor, var: bool, num_neighbour: int = None) -> torch.Tensor:
        """Calculates the mean prediction of the GP

        Args:
            testpoint (torch.Tensor): the input test point
            var (bool): will return the variance if True
            num_neighbour (int): the number of nearest neighbour to use for the BCM prediction

        Returns:
            torch.Tensor: the predicted function
        """
        test_trans = PreWhiten.x_transformation(self, testpoint)

        if self.exact:
            inputs = {'kernel': self.kernel, 'alpha': self.alpha,
                      'hyper': self.opt_parameters, 'xtrain': self.xtrain,
                      'ytrain': self.ytrain}

            if var:
                mean, variance = exact_predictions(test_trans, inputs, True)
                ypred = Normalisation.y_inv_transformation(self, mean)
                yvar = self.ystd**2 * variance
                return ypred.view(-1), yvar.view(-1)

            else:
                mean = exact_predictions(test_trans, inputs, False)
                ypred = Normalisation.y_inv_transformation(self, mean)
                return ypred.view(-1)
        else:
