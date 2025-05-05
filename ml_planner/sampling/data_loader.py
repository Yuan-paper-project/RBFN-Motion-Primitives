__author__ = "Marc Kaufed"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

import glob
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from ml_planner.general_utils.data_types import DTYPE
from ml_planner.general_utils.vehicle_models import STATE_LABELS


class DataSetStatesOnly(Dataset):
    """Custom Dataset for loading the dataset with OCP states and controls"""

    def __init__(self, dataset_dir: str):
        # load dataset
        dataset = glob.glob(f"{dataset_dir}/*.npz")
        assert len(dataset) == 1, f"Dataset not found or to many datasets in {dataset_dir}"
        with open(dataset_dir / "config.yaml", "r") as conf_file:
            self.dataparams = yaml.safe_load(conf_file)

        self.labels = STATE_LABELS[self.dataparams["vehicle"]["dynamic_model"]]

        with np.load(dataset[0]) as data:
            self.data = torch.from_numpy(data["state_vec"]).to(DTYPE)
            self.ocp_times = data["perf_data"]
        np.savetxt(dataset_dir / "perf_times.csv", self.ocp_times, delimiter=",")

        self.num_points_per_traj = self.data.shape[1]
        self.num_trajs = self.data.shape[0]

        self.input_labels = [i + "_0" for i in self.labels] + [i + "_f" for i in self.labels]
        self.bounds, self.bound_indices = get_limits(self)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # input vector: [initial_state, goal_state] of idx-th sample, including control values
        # [x0, y0, delta0, v0, psi0, ...,
        # xg, yg, vg, deltag, psig, ... ]

        # output vector: [complete trajectory states ] of idx-th sample
        try:
            input = torch.hstack([self.data[idx, 0], self.data[idx, -1]])
            output = self.data[idx]
        except IndexError:
            print(f"Index {idx} out of bounds")
            raise IndexError(f"Index {idx} out of bounds")
        return input, output


def get_limits(dataset: Dataset):
    """Returns the bounds  of the dataset and their indices"""
    all_mins, _ = dataset.data.min(dim=1)
    all_maxs, _ = dataset.data.max(dim=1)

    lb, lidx = all_mins.min(dim=0)
    ub, uidx = all_maxs.max(dim=0)

    bounds = torch.stack((lb, ub), dim=1)
    indices = torch.stack((lidx, uidx), dim=1)

    return bounds, indices  # , labels


def make_splits(dataset: Dataset, dataloader_kwargs: dict, labels: list[str] = None):
    """Splits the dataset into training and test dataset"""
    use_bounds_in_training = dataloader_kwargs.pop("use_bounds_in_training")
    train_split = dataloader_kwargs.pop("train_split")
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator)

    idxs = [dataset.labels.index(label) for label in labels] if labels is not None else None

    if use_bounds_in_training:
        # Ensure that bounds are in training dataset!
        bounds_idxs = set(dataset.bound_indices[idxs].unique().tolist())
        all_test_idxs = set(test_dataset.indices)
        all_train_idxs = set(train_dataset.indices)

        if not all_test_idxs.isdisjoint(bounds_idxs):
            test_idxs = all_test_idxs - bounds_idxs
            if len(test_idxs) == 0:
                test_idxs = set(list(all_train_idxs - bounds_idxs)[:1])
                all_train_idxs -= test_idxs
            train_idxs = all_train_idxs | bounds_idxs
            train_dataset.indices = list(train_idxs)
            test_dataset.indices = list(test_idxs)

    train_dataloader = DataLoader(train_dataset, **dataloader_kwargs)
    test_dataloader = DataLoader(test_dataset, **dataloader_kwargs)
    return train_dataloader, test_dataloader
