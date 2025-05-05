__author__ = "Marc Kaufed"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

import warnings
import shutil
import hydra
from pathlib import Path
from ml_planner.simulation_interfaces.commonroad.commonroad_interface import CommonroadInterface

"""
This script runs a simulation of a with a MP-RBFN Planner commonroad scenario.
"""

###############################
# PATH AND DEBUG CONFIGURATION
CWD = Path.cwd()
DATA_PATH = CWD / "example_scenarios"
SCENARIO = DATA_PATH / "ZAM_Tjunction-1_27_T-1.xml"
# SCENARIO = DATA_PATH / "ZAM_Over-1_1_dynamic_1vehicle_10m-s.xml"
LOG_PATH = CWD / "logs"

MODEL_PATH = CWD / "ml_planner" /  "sampling" / "models"

# debug configurations#
DELETE_ALL_FORMER_LOGS = False

LOGGING_LEVEL_INTERFACE = "debug"
LOGGING_LEVEL_PLANNER = "debug"

# Treat all RuntimeWarnings as errors
warnings.filterwarnings("error", category=RuntimeWarning)
###############################


def create_config():
    """
    Creates a configuration for the simulation.

    Returns:
        The configuration object for the simulation.
    """
    # config overrides
    overrides = [
        # general overrides
        f"log_path= {LOG_PATH}",
        # simulation overrides
        f"interface_logging_level={LOGGING_LEVEL_INTERFACE}",
        f"scenario_path={SCENARIO}",
        # planner overrides
        f"planner_config.logging_level={LOGGING_LEVEL_PLANNER}",
        f"planner_config.sampling_model_path={MODEL_PATH}",
    ]

    # Compose the configuration
    config_dir = str(Path.cwd() / "ml_planner" / "simulation_interfaces" / "commonroad" / "configurations")
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        config = hydra.compose(config_name="simulation", overrides=overrides)
    return config


def main():
    if DELETE_ALL_FORMER_LOGS:
        shutil.rmtree(LOG_PATH, ignore_errors=True)
    # create configuration
    config = create_config()

    # create simulation interface
    interface = CommonroadInterface(**config)
    # run simulation
    interface.run()
    # plot final trajectory
    interface.plot_final_trajectory()
    # make gif
    interface.create_gif()


if __name__ == "__main__":
    main()
    # TODO überarbeiten logging?
