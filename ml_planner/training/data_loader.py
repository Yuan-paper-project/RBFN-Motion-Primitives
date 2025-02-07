__author__ = "Marc Kaufed"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

import numpy as np
import re
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from ml_planner.planner.networks.utils import DTYPE


class DataSetAllLabels(Dataset):
    """Custom Dataset for loading the dataset with OCP states and controls"""
    def __init__(self, dataset_dir: str):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        # load dataset
        with np.load(dataset_dir) as dataset:
            self.data = torch.from_numpy(np.vstack([dataset[i] for i in dataset.files
                                                    ]))
        # read txt file of dataset with relevant information for the training
        txt_file_path = f"{dataset_dir._str[:-4]}.txt"
        info = ['dynamic_model', "vehicle_model", "labels", "planning_horizon", "dt", "dv", "dx", "dy", "dpsi"]
        self.dataparams = {}
        count = 0
        with open(txt_file_path, 'r') as file:
            for line in file:
                for i in info:
                    if count == len(info):
                        break
                    if i in line:
                        if i == "labels":
                            pattern = r'\'.*?\''
                            matches = re.findall(pattern, line)
                            self.labels = [match[1:-1] for match in matches]
                        elif i + ": " in line:
                            self.dataparams[i] = line.split(i + ": ")[1].split("\n")[0]
                        elif i + " = " in line:
                            self.dataparams[i] = float(line.split(i + " = ")[1].split("\n")[0])
                        count += 1
                        break

        self.input_labels = [i + "_0" for i in self.labels] + [i + "_f" for i in self.labels]
        self.bounds, self.bound_indices, self.bound_labels = get_limits(self)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # input vector: [initial_state, goal_state] of idx-th sample, including control values
        # [x0, y0, delta0, v0, psi0, ...,lat_control0, long_control0,
        # xg, yg, vg, deltag, psig, ... , lat_controlg, long_controlg]

        # output vector: [complete trajectory states  + control] of idx-th sample
        try:
            input = torch.hstack([self.data[idx][:, 0], self.data[idx][:, -1]])
            output = self.data[idx]
        except IndexError:
            print(f"Index {idx} out of bounds")
            raise IndexError(f"Index {idx} out of bounds")
        return input, output


def get_limits(dataset: Dataset):
    """Returns the bounds  of the dataset and their indices"""
    bounds_init = torch.tensor([[dataset.data[:, i, 0].min(dim=0, keepdim=True),
                                 dataset.data[:, i, 0].max(dim=0, keepdim=True)]
                                for i in range(dataset.data.shape[1])])
    labels_init = [i + "_0" for i in dataset.labels]
    bounds_goal = torch.tensor([[dataset.data[:, i, -1].min(dim=0, keepdim=True),
                                 dataset.data[:, i, -1].max(dim=0, keepdim=True)]
                                for i in range(dataset.data.shape[1])])
    labels_goal = [i + "_f" for i in dataset.labels]
    bounds = torch.vstack((bounds_init, bounds_goal))

    indices = bounds[:, :, 1].reshape(-1).to(int).tolist()
    bounds = bounds[:, :, 0].to(DTYPE)
    # expand zero width bounds
    mask = (bounds[:, 1] - bounds[:, 0]) < 0.1
    bounds[mask, 0] -= 1
    bounds[mask, 1] += 1
    labels = labels_init + labels_goal
    return bounds, indices, labels


def make_splits(dataset: Dataset, dataloader_kwargs: dict):
    """Splits the dataset into training and test dataset"""
    use_bounds_in_training = dataloader_kwargs.pop("use_bounds_in_training")
    train_split = dataloader_kwargs.pop("train_split")
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator)

    if use_bounds_in_training:
        # Ensure that bounds are in training dataset!
        bounds_idxs = set(dataset.bound_indices)
        all_test_idxs = set(test_dataset.indices)
        all_train_idxs = set(train_dataset.indices)

        if not all_test_idxs.isdisjoint(bounds_idxs):
            test_idxs = all_test_idxs - bounds_idxs
            train_idxs = all_train_idxs | bounds_idxs
            train_dataset.indices = list(train_idxs)
            test_dataset.indices = list(test_idxs)

    train_dataloader = DataLoader(train_dataset, **dataloader_kwargs)
    test_dataloader = DataLoader(test_dataset, **dataloader_kwargs)
    return train_dataloader, test_dataloader
