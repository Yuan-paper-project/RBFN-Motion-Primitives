__author__ = "Marc Kaufed"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

import os
from pathlib import Path
from typing import Callable
import torch

import ml_planner.planner.networks.rbf_network as networks
from ml_planner.planner.networks.loss_functions.loss_functions import PositionLoss, VelocityLoss, OrientationLoss
from ml_planner.training.model_trainer import ModelTrainer
from ml_planner.training.data_loader import DataSetAllLabels, make_splits
from ml_planner.planner.networks.layers.basis_functions import inverse_quadratic  # gaussian


def main():
    """Main Script to train the model on a OCP dataset"""

    # Training Data ###############
    # dataset configuration
    dataset_name = "dataset_KS_BMW320i_psi_full_complete.npz"
    dataset_dir = "dataset_KS_BMW320i_psi_full"

    dataset: Callable = DataSetAllLabels
    dataloader_kwargs: dict = {'train_split': .7,
                               "batch_size": 12000,
                               "use_bounds_in_training": True,
                               "shuffle": True,
                               "num_workers": 24,
                               "persistent_workers": True}

    # dataset loading
    cwd = Path.cwd()
    data_dir = cwd.parent / 'dataset' / dataset_dir / dataset_name

    dataset = dataset(data_dir)
    train_dataloader, test_dataloader = make_splits(dataset, dataloader_kwargs)

    # Model ###############
    # model configuration
    model_name = "SimpleRBF2"
    model_path = cwd / "ml_planner" / "planner" / "models" / model_name
    os.makedirs(model_path, exist_ok=True)

    model: Callable = networks.SimpleRBF
    # model: Callable = networks.ExtendedRBF
    # model: Callable = networks.MLP

    model_kwargs: dict = {"num_points_per_traj": train_dataloader.dataset.dataset.data.shape[2],
                          "input_labels": ['x_0', 'y_0', 'psi_0', 'v_0', 'x_f', 'y_f', 'psi_f'],
                          "output_labels": ['x', 'y', 'v', 'psi'],
                          "num_kernels": 512,
                          "basis_func": inverse_quadratic
                          }

    loss_functions = [PositionLoss, OrientationLoss, VelocityLoss]

    # model training
    trainer = ModelTrainer(train_dataloader=train_dataloader,
                           test_dataloader=test_dataloader,
                           model_path=model_path,
                           model=model,
                           model_kwargs=model_kwargs,
                           loss_functions=loss_functions,
                           optimizer=torch.optim.AdamW,
                           draw_trajectories_every_i_epoch=100
                           )

    trainer.run_training(epochs=7000)
    trainer.save_model('last')


if __name__ == "__main__":
    main()
