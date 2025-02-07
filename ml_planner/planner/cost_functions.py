__author__ = "Marc Kaufed"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

from abc import ABC
import torch
from ml_planner.planner.networks.utils import DTYPE


class Costs(ABC):

    def __init__(self, cost_weights, output_labels, device):
        super().__init__()
        # self.cost_weights = cost_weights
        self.device = device
        self.cost_terms = list(cost_weights.keys())
        self.output_labels = output_labels
        self.cost_functions = [getattr(self, term) for term in self.cost_terms
                               if callable(getattr(self, term, None))]
        self.cost_weights = torch.tensor([cost_weights[term] for term in self.cost_terms
                                          if callable(getattr(self, term, None))], dtype=DTYPE, device=self.device)

    def calculate(self, trajectories, reference_path, desired_velocity, predictions, boundary):
        """Calculate the total cost of a trajectory"""
        costs = [cost_function(trajectories, reference_path, desired_velocity, predictions, boundary)
                 for cost_function in self.cost_functions]
        costs = torch.stack(costs).T
        total_costs = costs @ self.cost_weights
        return total_costs, costs

    def acceleration(self, trajectories, reference_path, desired_velocity, predictions, boundary):
        """Calculate the acceleration cost along a trajectory"""
        try:
            idx = self.output_labels.index('acc')
            costs = trajectories[:, idx, :].pow(2).sum(dim=1)
        except ValueError:
            raise NotImplementedError
        return costs

    def distance_to_reference_path(self, trajectories, reference_path, desired_velocity, predictions, boundary):
        """Calculate the distance to the reference path
            Reference path is in vehicle coordinates -> distance only in lateral direction "y_loc"
        """
        idx_x = self.output_labels.index('x')
        idx_y = self.output_labels.index('y')
        path = reference_path
        costs = 0
        # along complete trajectory
        trajs = trajectories[:, [idx_x, idx_y], :].unsqueeze(1)
        path_expanded = path.unsqueeze(0).unsqueeze(-1)
        dist = torch.sqrt(torch.sum((trajs[:, :, [idx_x, idx_y], :] - path_expanded[:, :, :2, :])**2, dim=-2))

        dist_min, _ = dist.min(dim=1)
        costs += dist_min.sum(dim=1)

        # only at last position
        costs += dist_min[:, -1]*10
        return costs

    def velocity_offset(self, trajectories, reference_path, desired_velocity, predictions, boundary):
        """Calculate the velocity offset cost at end of trajectory"""
        costs = 0
        try:
            idx = self.output_labels.index('v')
            # at last position
            costs += (trajectories[:, idx, -1]-desired_velocity).pow(2)
        except ValueError:
            raise NotImplementedError
            # TODO calculate velocity from position -> maybe makes sense before Cost calculation to check this
        return costs

    def orientation_offset(self, trajectories, reference_path, desired_velocity, predictions, boundary):
        """Calculate the offset to the orientation of the reference path
            Reference path is in vehicle coordinates -> distance only in lateral direction "y_loc"
        """
        idx_x = self.output_labels.index('x')
        idx_y = self.output_labels.index('y')
        idx_psi = self.output_labels.index('psi')
        path = reference_path
        costs = 0

        # along complete trajectory
        trajs = trajectories[:, [idx_x, idx_y], :].unsqueeze(1)
        path_expanded = path.unsqueeze(0).unsqueeze(-1)
        dist = torch.sqrt(torch.sum((trajs[:, :, [idx_x, idx_y], :] - path_expanded[:, :, :2, :])**2, dim=-2))
        mask = dist.argmin(dim=1, keepdim=False)
        costs += torch.sum((trajectories[:, idx_psi, :] - path[mask][:, :, 2])**2, dim=1)

        # only at last position
        path_expanded = path.unsqueeze(0).expand(trajectories.shape[0], -1, -1)
        dist = torch.sqrt(torch.sum((trajs[:, :, [idx_x, idx_y], -1] - path_expanded[:, :, :2])**2, dim=-1))
        mask = dist.argmin(dim=1, keepdim=False)
        costs += (trajectories[:, idx_psi, -1] - path[mask][:, 2]).pow(2)*20
        return costs

    def prediction(self, trajectories, reference_path, desired_velocity, predictions, boundary):
        """Calculate the prediction costs based on the inverse mahalanobis distance"""
        if predictions:
            x_idx = self.output_labels.index('x')
            y_idx = self.output_labels.index('y')
            indices = [x_idx, y_idx]
            pred_pos = predictions[0]
            pred_cov = predictions[1]
            inv_cov = torch.inverse(pred_cov)

            pred_pos = predictions[0]
            pred_cov = predictions[1]
            inv_cov = torch.inverse(pred_cov)
            traj = trajectories[:, indices, :pred_pos.shape[-2]]
            traj = traj.transpose(-1, -2).unsqueeze(1)

            delta = (traj - pred_pos).unsqueeze(-1)
            mahalanobis_square = delta.transpose(-1, -2) @ inv_cov  @ delta
            inv_dist = 1 / (mahalanobis_square.squeeze(-1).squeeze(-1))
            costs = inv_dist.sum(dim=-1).sum(dim=-1)

        else:
            costs = torch.zeros(trajectories.shape[0], dtype=DTYPE, device=self.device)
        return costs

    def distance_to_boundary(self, trajectories, reference_path, desired_velocity, predictions, boundary):
        """Calculate the cost based on the distance to the boundary"""

        idx_x = self.output_labels.index('x')
        idx_y = self.output_labels.index('y')
        indices = [idx_x, idx_y]

        # along complete trajectory
        trajs = trajectories[:, indices, :].unsqueeze(1)
        bounds_extended = boundary.unsqueeze(0).unsqueeze(-1)

        dist = torch.sqrt(torch.sum((trajs[:, :, indices, :] - bounds_extended[:, :, :2, :])**2, dim=-2))

        dist_min, _ = dist.min(dim=1, keepdim=False)
        costs = (1 / dist_min**2).sum(dim=1)

        return costs
