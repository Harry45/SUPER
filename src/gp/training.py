# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: In this code, we train the GP model using the training data.

import os
import torch
import matplotlib.pylab as plt

# our script and functions
import utils.helpers as hp
import config as CONFIG

plt.rc("text", usetex=True)
plt.rc("font", **{"family": "sans-serif", "serif": ["Palatino"]})
figSize = (12, 8)
fontSize = 20


def plot_loss(optim: dict, fname: str, save: bool = True):
    """Plots the loss of the GP model.

    Args:
        optim (dict): A dictionary with the optimizer. There can be more than 1 optimisation.
        save (bool, optional): Whether to save the plot. Defaults to True.
    """

    nopt = len(optim)
    niter = len(optim[0]["loss"])

    # plot the loss
    plt.figure(figsize=(8, 8))
    for i in range(nopt):
        plt.plot(range(niter), optim[i]["loss"], label=f"Optimization {i + 1}")
    plt.xlabel("Iterations", fontsize=fontSize)
    plt.ylabel("Loss", fontsize=fontSize)
    plt.tick_params(axis="x", labelsize=fontSize)
    plt.tick_params(axis="y", labelsize=fontSize)
    plt.legend(loc="best", prop={"family": "sans-serif", "size": 15})

    path = os.path.join("results", "loss")
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + "/" + fname + ".pdf", bbox_inches="tight")
    if save:
        plt.close()
    else:
        plt.show()


def train_gps(nlhs: int, xtrans: bool, ytrans: bool, jitter: float = 1e-6) -> list:
    """Train the Gaussian Processes and store them.

    Args:
        nlhs (int): number of Latin Hypercube samples.
        xtrans (bool): the transformation for the inputs.
        ytrans (bool): the transformation for the outputs.
        jitter (float, optional): the jitter term for numerical stability. Defaults to 1E-6.

    Returns:
        list: A list of Gaussian Processes.
    """

    inputs = hp.load_csv("data", "cosmologies_lhs_" + str(nlhs))
    outputs = hp.load_csv("data", "pk_linear_lhs_" + str(nlhs))

    ins = torch.from_numpy(inputs.values)

    gps = list()

    for i in range(CONFIG.NWAVE):

        print(f"Training GP {i + 1}")

        out = torch.from_numpy(outputs.iloc[:, i].values)

        # the GP module
        gp_module = GaussianProcess(inputs, targets, induce, jitter)

        # perform the optimisation of the GP model
        opt = gp_module.optimisation(
            torch.randn(6),
            niter=CONFIG.NITER,
            lrate=CONFIG.LEARN_RATE,
            nrestart=CONFIG.NRESTART,
        )

        gps.append(gp_module)

        # plot and store the loss function of the GP model
        plot_loss(opt, f"{str(nlhs)}/pk_linear_lhs_" + str(nlhs) + "_wave_" + str(i))

        # save the GP model
        path = "gps/" + str(nlhs)
        os.makedirs(path, exist_ok=True)

        # name of the files to save
        gp_name = "pk_linear_lhs_" + str(nlhs) + "_wave_" + str(i)
        pa_name = "params_" + str(nlhs) + "_wave_" + str(i)
        al_name = "alpha_" + str(nlhs) + "_wave_" + str(i)

        hp.save_list(gp_module, path, gp_name)
        hp.save_list(gp_module.opt_parameters.data.numpy(), path, pa_name)
        hp.save_list(gp_module.alpha.data.numpy(), path, al_name)

    return gps
