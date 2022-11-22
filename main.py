"""
Authors: Arrykrishna Mootoovaloo
Email: arrykrish@gmail.com
Date: November 2022
Project: Implementation of a scalable GP approach for emulating power spectra
Script: Main script for running the code
"""

from absl import flags, app
from ml_collections.config_flags import config_flags

# our scripts
from emulator.trainingpoints import scale_lhs
from src.cosmo.matterpower import class_compute
from utils.checkers import check_config, make_paths
from utils.logger import get_logger


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Main configuration file.", lock_config=True)


def main(argv):
    """
    Run the main script.
    """
    check_config(FLAGS.config)
    make_paths(FLAGS.config)

    logger = get_logger(FLAGS.config, 'main')
    logger.info("Running main script")

    # cosmologies = scale_lhs(FLAGS.config, 'lhs_1000', True, fname='1000')
    cosmo = {'omega_cdm': 0.12, 'omega_b': 0.022, 'ln10^{10}A_s': 3.3, 'n_s': 1.0, 'h': 0.70}
    module = class_compute(FLAGS.config, cosmo)


if __name__ == "__main__":
    app.run(main)
