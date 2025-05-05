__author__ = "Marc Kaufed"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

import os
import warnings
from pathlib import Path
from typing import Callable
import torch

import ml_planner.sampling.networks.rbf_network as networks
from ml_planner.sampling.networks.layers.loss_functions import PositionLoss, VelocityLoss, OrientationLoss, SteeringLoss
from ml_planner.sampling.model_trainer import ModelTrainer
from ml_planner.sampling.data_loader import DataSetStatesOnly, make_splits


###############################
# PATH AND DEBUG CONFIGURATION
CWD = Path.cwd()
DATA_PATH = CWD.parent / "dataset"
LOG_PATH = CWD / "logs"


# debug configurations#
DELETE_ALL_FORMER_LOGS = False

LOGGING_LEVEL = "debug"

# Treat all RuntimeWarnings as errors
warnings.filterwarnings("error", category=RuntimeWarning)
###############################


def main():
    """Main Script to train the model on a OCP dataset"""

    # Training Data ###############
    # dataset configuration
    dataset_name = "dataset_BMW320i_kinematic_single_track_steering_jerk_full_v1"
    dataset_dir = DATA_PATH

    dataset: Callable = DataSetStatesOnly
    dataloader_kwargs: dict = {
        "train_split": 0.7,
        "batch_size": 12000,
        "use_bounds_in_training": True,
        "shuffle": True,
        "num_workers": 10,
        "persistent_workers": True,
    }

    # dataset loading
    data_dir = dataset_dir / dataset_name
    dataset = dataset(data_dir)

    # Model ###############
    # model configuration
    model_name = "extended_rbf_woInt_gaussian_512_kinematic_single_track_steering_jerk_wo_acc_w_delta"
    model_path = CWD / "ml_planner" / "sampling" / "models" / model_name
    os.makedirs(model_path, exist_ok=True)

    # model: Callable = networks.SimpleRBF
    # model: Callable = networks.ExtendedRBF
    model: Callable = networks.ExtendedRBF_woInt
    # model: Callable = networks.MLP1Layer

    model_kwargs: dict = {
        "num_points_per_traj": dataset.num_points_per_traj,
        "input_labels": ["x_0", "y_0", "psi_0", "v_0", "delta_0", "x_f", "y_f", "psi_f"],
        "output_labels": ["x", "y", "psi", 'v', "delta"],
        "num_kernels": 512,
        "basis_func": "gaussian",  # "gaussian" "inverse_quadratic", "inverse_multiquadric", "multiquadric", "spline", "poisson_one", "poisson_two", "matern32", "matern52"
        "bounds": dataset.bounds,
    }

    loss_functions = [PositionLoss, OrientationLoss, VelocityLoss, SteeringLoss]

    # make train and test splits
    train_dataloader, test_dataloader = make_splits(dataset, dataloader_kwargs, model_kwargs["output_labels"])

    # model training
    trainer = ModelTrainer(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        model_path=model_path,
        model=model,
        model_kwargs=model_kwargs,
        # device="cpu",
        loss_functions=loss_functions,
        optimizer=torch.optim.AdamW,
        draw_trajectories_every_i_epoch=500,
    )

    trainer.run_training(epochs=2000)
    trainer.save_model("last")


if __name__ == "__main__":
    main()
