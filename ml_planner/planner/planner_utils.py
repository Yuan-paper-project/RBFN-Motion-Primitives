__author__ = "Marc Kaufed"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

import json
import torch
import numpy as np

from ml_planner.planner.networks.utils import DTYPE
from .networks.layers.basis_functions import basis_func_dict
import ml_planner.planner.networks.rbf_network as networks
from ml_planner.helper_functions.data_types import StateVector


def load_model(model_path, model_type, device):
    # load config
    config_file = model_path / "config.json"
    with open(config_file, 'r') as json_file:
        model_kwargs = json.load(json_file)
    dt = model_kwargs.pop('dt')
    planning_horizon = model_kwargs.pop('planning_horizon')
    info = {'dv': model_kwargs.pop('dv'),
            'dx': model_kwargs.pop('dx'),
            'dy': model_kwargs.pop('dy'),
            'dpsi': model_kwargs.pop('dpsi'),
            'dynamic_model': model_kwargs.pop('dynamic_model'),
            'vehicle_model': model_kwargs.pop('vehicle_model')
            }
    # instantiate model
    model_kwargs['basis_func'] = basis_func_dict()[model_kwargs['basis_func']]
    model_cls = getattr(networks, model_type)
    model = model_cls(**model_kwargs)
    # load weights
    model_weigths_file = [str(file) for file in model_path.glob("*best.pth") if file.is_file()][-1]
    model.load_state_dict(torch.load(model_weigths_file, weights_only=True, map_location=device))
    return model, info, dt, planning_horizon


def convert_globals_to_locals(states: torch.Tensor, center: StateVector):
    """
    Converts global coordinates to local reference frame
    state & center: [x,y,psi]
    """
    # states can be a list of states or a list of list of states
    states = states.reshape(-1, states.shape[-2], states.shape[-1])
    rot_matrix = torch.tensor([[np.cos(center.psi), np.sin(center.psi)],
                               [-np.sin(center.psi), np.cos(center.psi)]], dtype=DTYPE, device=states.device)
    transl = states[:, :, :2] - torch.tensor([center.x, center.y], dtype=DTYPE, device=states.device)

    new_states = transl @ rot_matrix.T
    if states.shape[-1] == 3:
        # if states are [x,y,psi]
        psi = (states[:, :, -1] - center.psi).unsqueeze(-1)
        new_states = torch.cat((new_states, psi), dim=2)
    return new_states.squeeze()


def convert_locals_to_globals(states: torch.Tensor, center: StateVector):
    """
    Converts local coordinates to global reference frame
    state & center: [x,y,psi]
    """
    # states can be a list of states or a list of list of states
    states = states.reshape(-1, states.shape[-2], states.shape[-1])
    rot_matrix = torch.tensor([[np.cos(center.psi), -np.sin(center.psi)],
                               [np.sin(center.psi), np.cos(center.psi)]], dtype=DTYPE, device=states.device)
    transl = states[:, :, :2]
    new_states = transl @ rot_matrix.T
    new_states += torch.tensor([center.x, center.y], dtype=DTYPE, device=states.device)
    if states.shape[-1] == 3:
        # if states are [x,y,psi]
        psi = (states[:, :, -1] + center.psi).unsqueeze(-1)
        new_states = torch.cat((new_states, psi), dim=2)
    return new_states.squeeze()


def rotate_covariance_matrix(cov_matrix, center: StateVector):
    # Rotate the covariance matrix of the predicitons
    rot_matrix = torch.tensor([[np.cos(center.psi), np.sin(center.psi)],
                               [-np.sin(center.psi), np.cos(center.psi)]], dtype=DTYPE, device=cov_matrix.device)
    rotated_cov_matrix = rot_matrix @ cov_matrix @ rot_matrix.T
    return rotated_cov_matrix
