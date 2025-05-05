__author__ = "Marc Kaufeld"
__copyright__ = "TUM Professorship Autonomous Vehicle Systems"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

import torch
import torch.nn as nn
from typing import Optional, Union


DTYPE = torch.float32


# @dataclass(kw_only=True)
class StateTensor(nn.Module):  # ABC):

    def __init__(
        self, states: Union[torch.Tensor, dict], device: str, covs: Optional[Union[torch.Tensor, dict]] = None
    ):
        super().__init__()

        self.state_labels = ["x", "y", "psi", "v", "delta"]
        self.device = device
        self.x_idx = self.state_labels.index("x")
        self.y_idx = self.state_labels.index("y")
        self.psi_idx = self.state_labels.index("psi")
        self.v_idx = self.state_labels.index("v")
        self.delta_idx = self.state_labels.index("delta")

        if isinstance(states, dict):
            assert (
                list(states.keys()) == self.state_labels[: len(list(states.keys()))]
            ), "State dict keys must match state label sorting "
            states = torch.stack(
                [torch.tensor(states[i], dtype=DTYPE) for i in self.state_labels if i in states], dim=-1
            )
            if covs:
                assert list(covs.keys()) == list(states.keys()), "Cov dict keys must match state label sorting "
                covs = torch.stack(
                    [torch.tensor(covs[i], dtype=DTYPE) for i in self.state_labels if i in states], dim=-1
                )
        assert (
            states.dim() <= 3
        ), "States must be 1D (single state), 2D (list of states) or 3D tensor (batch of list of states)"
        self.states: torch.Tensor = states.to(device)
        self.covs = None
        if covs is not None:
            self.add_cov(covs)

    def extra_repr(self) -> str:
        return (
            f"""States: {self.state_variables} \nNumber of diff states: {self.num_states}"""
            f"""\nwith covs: {True if self.covs is not None else False}"""
        )

    def add_cov(self, covs: torch.Tensor):
        assert (
            covs.dim() == self.states.dim() + 1
        ), "Covariances must be 2D (single state), 3D (list of states) or 4D tensor (batch of list of states)"

        self.covs: torch.Tensor = covs.to(self.device)

    @property
    def state_variables(self):
        return self.state_labels[: self.states.shape[-1]]

    @property
    def num_variables(self):
        return self.states.shape[-1]

    @property
    def num_states(self):
        if self.states.ndim == 1:
            return 1

        # else:
        return self.states.shape[-2]

    @property
    def num_batches(self):
        if self.states.ndimension() == 3:
            return self.states.shape[0]
        return 1

    @property
    def coords(self):
        # coordinates
        return self.states[..., [self.x_idx, self.y_idx]]

    @property
    def coords_cov(self):
        return (
            self.covs[..., [self.x_idx, self.y_idx], :][..., [self.x_idx, self.y_idx]]
            if self.covs is not None
            else None
        )

    @property
    def psi(self):
        # orientation
        try:
            return self.states[..., self.psi_idx]
        except IndexError:
            return None

    @property
    def v(self):
        # velocity
        try:
            return self.states[..., self.v_idx]
        except IndexError:
            return None

    @property
    def delta(self):
        # steering angle
        try:
            return self.states[..., self.delta_idx]
        except IndexError:
            return None

    def sort_indices(self, mask):
        self.states = self.states[mask]
        if self.covs is not None:
            self.covs = self.covs[mask]

    def detach(self):
        states = self.states.detach()
        covs = self.covs.detach() if self.covs is not None else None
        return StateTensor(states, self.device, covs)
