__author__ = "Marc Kaufed"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

from typing import Callable
import torch
from torch import nn

from .layers.rbf_layer import RBFLayer1D, RBFLayernD
from .utils import DTYPE


class ExtendedRBF(nn.Module):
    """RBF network with linear interpolation"""
    def __init__(self,
                 input_labels: list[str],
                 output_labels: list[str],
                 num_points_per_traj: int,
                 num_kernels: int,
                 basis_func: Callable):

        super().__init__()
        # network dimensions
        self.input_labels = input_labels
        self.input_dim = len(input_labels)
        self.output_labels = output_labels
        self.output_dim = len(output_labels) * (num_points_per_traj)
        self.num_points_per_traj = num_points_per_traj
        self.num_kernels = num_kernels

        # layers
        self.input_layer = nn.Linear(in_features=self.input_dim,
                                     out_features=self.num_kernels,
                                     bias=False)
        self.rbflayer = RBFLayer1D(num_kernels=self.num_kernels,
                                   basis_func=basis_func)
        self.linear = nn.Linear(in_features=self.num_kernels,
                                out_features=self.output_dim,
                                bias=False)

    def forward(self, input):
        """Forward pass"""
        y = self.input_layer(input)
        y = self.rbflayer(y)
        pred = self.linear(y)
        traj = self.pred_to_trajectory(pred)
        pos_base = self.linear_interpolation(input)
        output = traj + pos_base
        return output

    def extra_repr(self) -> str:
        txt = f"in_features={self.input_dim}, out_features={self.output_dim}"
        return txt

    def linear_interpolation(self, x):
        """Linear interpolation between initial and final values"""
        x0 = torch.zeros([x.shape[0], len(self.output_labels), 1], dtype=DTYPE, device=x.device)
        xf = torch.zeros_like(x0, dtype=DTYPE, device=x.device)
        x0_idxs = [self.input_labels.index(i + "_0") for i in self.output_labels if i + "_0" in self.input_labels]
        x0[:, sorted(x0_idxs)] = x[:, x0_idxs].unsqueeze(-1)
        xf_idxs = [self.input_labels.index(i + "_f") if i + "_f" in self.input_labels
                   else self.input_labels.index(i + "_0") for i in self.output_labels if i + "_0" in self.input_labels]
        xf[:, sorted(x0_idxs)] = x[:, xf_idxs].unsqueeze(-1)

        steps = self.num_points_per_traj
        line = torch.linspace(0, 1, steps, device=x.device).unsqueeze(0).unsqueeze(0)
        base = x0 * (1 - line) + xf * line
        return base

    def pred_to_trajectory(self, x):
        """Reshape the output to a trajectory"""
        return x.reshape(-1,  len(self.output_labels), self.num_points_per_traj)


class ExtendedRBF_woInt(nn.Module):
    """RBF network without linear interpolation"""
    def __init__(self,
                 input_labels: list[str],
                 output_labels: list[str],
                 num_points_per_traj: int,
                 num_kernels: int,
                 basis_func: Callable):

        super().__init__()
        # network dimensions
        self.input_labels = input_labels
        self.input_dim = len(input_labels)
        self.output_labels = output_labels
        self.output_dim = len(output_labels) * (num_points_per_traj)
        self.num_points_per_traj = num_points_per_traj
        self.num_kernels = num_kernels

        # layers
        self.input_layer = nn.Linear(in_features=self.input_dim,
                                     out_features=self.num_kernels,
                                     bias=False)
        self.rbflayer = RBFLayer1D(num_kernels=self.num_kernels,
                                   basis_func=basis_func)
        self.linear = nn.Linear(in_features=self.num_kernels,
                                out_features=self.output_dim,
                                bias=False)

    def forward(self, input):
        """Forward pass"""
        y = self.input_layer(input)
        y = self.rbflayer(y)
        pred = self.linear(y)
        traj = self.pred_to_trajectory(pred)
        output = traj
        return output

    def extra_repr(self) -> str:
        txt = f"in_features={self.input_dim}, out_features={self.output_dim}"
        return txt

    def pred_to_trajectory(self, x):
        """Reshape the output to a trajectory"""
        return x.reshape(-1,  len(self.output_labels), self.num_points_per_traj)


class SimpleRBF(nn.Module):
    """Simple (classic) RBF network"""
    def __init__(self,
                 input_labels: list[str],
                 output_labels: list[str],
                 num_points_per_traj: int,
                 num_kernels: int,
                 basis_func: Callable):

        super().__init__()
        # network dimensions
        self.input_labels = input_labels
        self.input_dim = len(input_labels)
        self.output_labels = output_labels
        self.output_dim = len(output_labels) * (num_points_per_traj)
        self.num_points_per_traj = num_points_per_traj
        self.num_kernels = num_kernels

        # layers
        self.rbflayer = RBFLayernD(in_features=self.input_dim, out_features=self.num_kernels,
                                   basis_func=basis_func)
        self.linear = nn.Linear(in_features=self.num_kernels,
                                out_features=self.output_dim,
                                bias=False)

    def forward(self, input):
        """Forward pass"""
        y = self.rbflayer(input)
        pred = self.linear(y)
        traj = self.pred_to_trajectory(pred)
        output = traj
        return output

    def extra_repr(self) -> str:
        txt = f"in_features={self.input_dim}, out_features={len(self.output_labels)}x{self.num_points_per_traj}"
        return txt

    def pred_to_trajectory(self, x):
        """Reshape the output to a trajectory"""
        return x.reshape(-1,  len(self.output_labels), self.num_points_per_traj)


class MLP(nn.Module):
    """Simple MLP network"""
    def __init__(self,
                 input_labels: list[str],
                 output_labels: list[str],
                 num_points_per_traj: int,
                 num_kernels: int,
                 basis_func: Callable):

        super().__init__()
        # network dimensions
        self.input_labels = input_labels
        self.input_dim = len(input_labels)
        self.output_labels = output_labels
        self.output_dim = len(output_labels) * num_points_per_traj
        self.num_points_per_traj = num_points_per_traj
        self.num_kernels = num_kernels

        # layers
        self.input_layer = nn.Linear(self.input_dim, self.num_kernels, bias=False)
        self.activation1 = nn.ReLU()
        self.linear = nn.Linear(self.num_kernels, self.num_kernels, bias=False)
        self.activation2 = nn.ReLU()
        self.output_layer = nn.Linear(self.num_kernels, self.output_dim, bias=False)

    def forward(self, input):
        """forward pass"""
        y = self.input_layer(input)
        y = self.activation1(y)
        y = self.linear(y)
        y = self.activation2(y)
        output = self.output_layer(y)
        output = self.pred_to_trajectory(output)
        return output

    def extra_repr(self) -> str:
        txt = f"in_features={self.input_dim}, out_features={self.output_dim}"
        return txt

    def pred_to_trajectory(self, x):
        """Reshape the output to a trajectory"""
        return x.reshape(-1,  len(self.output_labels), self.num_points_per_traj)
