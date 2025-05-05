__author__ = "Marc Kaufed"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

import torch
import torch.nn as nn
from typing import Callable
from ml_planner.general_utils.data_types import DTYPE

class RBFLayernD(nn.Module):
    """
    Transforms incoming data using a given radial basis function:
    u_{i} = rbf(||x - c_{i}|| / s_{i})

    Arguments:
        in_features: size of each input sample
        out_features: size of each output sample

    Shape:
        - Input: (N, in_features) where N is an arbitrary batch size
        - Output: (N, out_features) where N is an arbitrary batch size

    Attributes:
        centres: the learnable centres of shape (out_features, in_features).
            The values are initialised from a standard normal distribution.
            Normalising inputs to have mean 0 and standard deviation 1 is
            recommended.

        log_sigmas: logarithm of the learnable scaling factors of shape (out_features).

        basis_func: the radial basis function used to transform the scaled
            distances.
    """

    def __init__(self, in_features: int, out_features: int, basis_func: Callable):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centers = nn.Parameter(torch.empty(self.in_features, self.out_features, dtype=DTYPE))
        self.eps = nn.Parameter(torch.empty(self.out_features, dtype=DTYPE))
        self.basis_func = basis_func
        self.reset_parameters()

    def extra_repr(self) -> str:
        return f"""in_features={self.in_features},
                num_kernels/out_features={self.out_features},
                basis func={self.basis_func.__name__}"""

    def reset_parameters(self):
        generator = torch.Generator().manual_seed(42)
        nn.init.uniform_(self.centers, 0, 10, generator)
        nn.init.constant_(self.eps, 1)

    def forward(self, input):
        x = input.unsqueeze(-1)
        distances = (x - self.centers).pow(2).sum(1).pow(0.5) * self.eps
        output = self.basis_func(distances)
        self.distance = distances  # to log result in tensorboard
        self.output = output  # to log result in tensorboard
        return output


class RBFLayer1D(nn.Module):
    """
    Transforms incoming data using a given radial basis function:
    u_{i} = rbf(||x - c_{i}|| / s_{i})

    Arguments:
        in_features: size of each input sample
        out_features: size of each output sample

    Shape:
        - Input: (N, in_features) where N is an arbitrary batch size
        - Output: (N, out_features) where N is an arbitrary batch size

    Attributes:
        centres: the learnable centres of shape (out_features, in_features).
            The values are initialised from a standard normal distribution.
            Normalising inputs to have mean 0 and standard deviation 1 is
            recommended.

        log_sigmas: logarithm of the learnable scaling factors of shape (out_features).

        basis_func: the radial basis function used to transform the scaled
            distances.
    """

    def __init__(self, num_kernels: int, basis_func: Callable):
        super().__init__()

        self.num_kernels = num_kernels
        self.centers = nn.Parameter(torch.empty(self.num_kernels, dtype=DTYPE))
        self.eps = nn.Parameter(torch.empty(num_kernels, dtype=DTYPE))
        self.basis_func = basis_func
        self.distance = None
        self.output = None
        self.reset_parameters()

    def extra_repr(self) -> str:
        return f"num_kernels/out_features={self.num_kernels}, basis func={self.basis_func.__name__}"

    def reset_parameters(self):
        generator = torch.Generator().manual_seed(42)
        nn.init.uniform_(self.centers, 0, 10, generator)
        nn.init.constant_(self.eps, 1)

    def forward(self, x):
        distances = (x - self.centers) * self.eps
        output = self.basis_func(distances)
        self.distance = distances  # to log result in tensorboard
        self.output = output  # to log result in tensorboard
        return output
