from torch.multiprocessing import Pool, set_start_method

try:
    set_start_method('spawn')
except RuntimeError:
    pass


def log_marginal_likelihood_parallel(args: list) -> list:
    """Calculates the log marginal likelihood in parallel.

    Args:
        args (list): a list of list consisting of the kernel, targets and jitter term.

    Returns:
        list: a list of the marginal likelihood due to each partition.
    """
    kernel = args[0]
    targets = args[1]
    jitter = args[2]
    log_ml = log_marginal_likelihood(kernel, targets, jitter)
    return log_ml


def cost_parallel(record: dict, hyperparameters: torch.Tensor, jitter: float) -> torch.Tensor:
    """Calculates the total cost given a dictionary consisting of the
    partitioned data and the hyperparameters.

    To Improve - need to check how to get multiprocessing working with PyTorch.

    Args:
        record (dict): A dictionary containing the partitioned data
        hyperparameters (torch.Tensor): The kernel hyperparameter.
        jitter (float): The jitter term for numerical stability

    Returns:
        torch.Tensor: The total cost
    """
    nclusters = len(record)
    arguments = list()
    for i in range(nclusters):
        xpoint, ypoint = record[str(i)][0], record[str(i)][1]
        kernel = compute_kernel(xpoint, xpoint, hyperparameters)
        arguments.append([kernel, ypoint, jitter])

    nprocesses = torch.multiprocessing.cpu_count()
    pool = Pool(processes=nprocesses)
    costs = pool.map(log_marginal_likelihood_parallel, arguments)
    pool.close()
    pool.join()

    total_cost = torch.FloatTensor(costs).sum(0).detach()

    return total_cost
