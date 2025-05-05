__author__ = "Marc Kaufed"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"


import torch
from ml_planner.general_utils.data_types import DTYPE, StateTensor 


def convert_globals_to_locals(states: StateTensor, center: StateTensor):
    """
    Converts global coordinates to local reference frame
    states: [batch, seq_len, (x,y,psi, ...)]
    center: [x, y, psi, (v,...)]
    """
    rot_matrix = torch.eye(states.num_variables, dtype=DTYPE, device=center.device)
    rot_matrix[:2, :2] = torch.tensor(
        [
            [torch.cos(center.psi), -torch.sin(center.psi)],
            [torch.sin(center.psi), torch.cos(center.psi)],
        ],
        dtype=DTYPE,
        device=center.device,
    )
    shift = torch.zeros(center.num_variables, dtype=DTYPE, device=center.device)
    shift[[center.x_idx, center.y_idx, center.psi_idx]] = 1
    new_states = (states.states - (center.states * shift)[: states.num_variables]) @ rot_matrix

    if states.covs is not None:
        num_states = states.covs.shape[-1]
        covs = rot_matrix[:num_states, :num_states].T @ states.covs @ rot_matrix[:num_states, :num_states]
    else:
        covs = None
    new_states = StateTensor(states=new_states, covs=covs, device=states.device)
    return new_states


def convert_locals_to_globals(states: StateTensor, center: StateTensor):
    """
    Converts local coordinates to global reference frame
    state & center: [x,y,psi]
    """

    rot_matrix = torch.eye(states.num_variables, dtype=DTYPE, device=center.device)
    rot_matrix[:2, :2] = torch.tensor(
        [
            [torch.cos(center.psi), torch.sin(center.psi)],
            [-torch.sin(center.psi), torch.cos(center.psi)],
        ],
        dtype=DTYPE,
        device=center.device,
    )
    shift = torch.zeros(center.num_variables, dtype=DTYPE, device=center.device)
    shift[[states.x_idx, states.y_idx, states.psi_idx]] = 1
    new_states = states.states @ rot_matrix + (center.states * shift)[: states.num_variables]

    if states.covs is not None:
        num_states = states.covs.shape[-1]
        covs = rot_matrix[:num_states, :num_states].T @ states.covs @ rot_matrix[:num_states, :num_states]
    else:
        covs = None
    new_states = StateTensor(states=new_states, covs=covs, device=states.device)
    return new_states


def create_cov_matrix(curve: StateTensor, standard_devs: dict):
    """create a tensor with covariance matrices for each state based on the uncertainties dict"""
    keys = [key for key in standard_devs.keys() if key[:2] != "d_"]
    num_cov_states = len(keys)
    assert (
        keys == curve.state_variables[:num_cov_states]
    ), "standard_devs keys must match the state variables of the curve"
    stds = torch.tensor([val for key, val in standard_devs.items() if key in curve.state_variables])
    d_stds = torch.tensor(
        [val for key, val in standard_devs.items() if key[:2] == "d_" and key[2:] in curve.state_variables]
    )

    cov = torch.diag(stds).repeat(curve.num_states, 1, 1)
    if len(d_stds):
        ind = torch.arange(num_cov_states)
        increment = torch.arange(curve.num_states).view(-1, 1) * d_stds
        cov[:, ind, ind] += increment
    cov = cov**2  # covariance is square of the standard deviation
    return cov.squeeze()
