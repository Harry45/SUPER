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
from emulator.trainingpoints import scale_lhs, generate_training_pk
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

    logger = get_logger(FLAGS.config)
    logger.info("Running main script")

    cosmologies = scale_lhs(FLAGS.config, 'lhs_5d_1000', True, fname='5d_1000')
    powerspectra = generate_training_pk(FLAGS.config, fname='5d_1000')


if __name__ == "__main__":
    app.run(main)
