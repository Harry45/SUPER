"""
Authors: Arrykrishna Mootoovaloo
Email: arrykrish@gmail.com
Date: November 2022
Project: Implementation of a scalable GP approach for emulating power spectra
Script: The logging file to store all the logs.
"""

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from ml_collections.config_dict import ConfigDict

NOW = datetime.now()
FORMATTER = logging.Formatter("[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d")
CONSOLE_FORMATTER = logging.Formatter("[%(levelname)s]: %(message)s")
DATETIME = NOW.strftime("%d-%m-%Y-%H-%M")


def get_logger(config: ConfigDict, name: str) -> logging.Logger:
    """Generates a logging file for storing all information.

    Args:
        config (ConfigDict): The main configuration file
        name (str): specific name for the recording the logs

    Returns:
        logging.Logger: the logging module
    """
    fname = config.path.logs + config.logname + f'_{DATETIME}.log'
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    chandler = logging.StreamHandler(sys.stdout)
    chandler.setLevel(logging.INFO)
    chandler.setFormatter(CONSOLE_FORMATTER)

    fhandler = logging.FileHandler(filename=fname, mode='a')
    fhandler.setFormatter(FORMATTER)

    logger.addHandler(fhandler)
    logger.addHandler(chandler)
    return logger
