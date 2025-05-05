__author__ = "Marc Kaufed"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

import torch
import torch.nn as nn
from ml_planner.general_utils.data_types import DTYPE


class Costs(nn.Module):

    def __init__(self, cost_weights, device):
        super().__init__()
        self.device = device
        self.cost_terms = [cost for cost, val in cost_weights.items() if callable(getattr(self, cost, None)) and val != 0]
        self.cost_functions = [getattr(self, term) for term in self.cost_terms]
        self.cost_weights = nn.Parameter(
            torch.tensor(
                [cost_weights[term] for term in self.cost_terms],
                dtype=DTYPE,
                device=self.device,
            )
        )

    def extra_repr(self) -> str:
        return f"""cost_terms={self.cost_terms},"""

    def forward(self, trajectories, ego_shape,  reference_path, desired_velocity, obstacle_predictions, lane_predictions):
        """Calculate the total cost of a trajectory"""
        costs = [
            cost_function(trajectories, ego_shape, reference_path, desired_velocity, obstacle_predictions, lane_predictions)
            for cost_function in self.cost_functions
        ]
        costs = torch.stack(costs).T
        total_costs = costs @ self.cost_weights
        return total_costs, costs

    def distance_to_reference_path(
        self, trajectories, ego_shape, reference_path, desired_velocity, obstacle_predictions, lane_predictions
    ):
        """Calculate the mahalanobis distance to the reference path
        based on own state uncertainty
        Reference path is in vehicle coordinates
        """
        path = reference_path.coords
        traj = trajectories.coords
        inv_cov = torch.inverse(trajectories.coords_cov).unsqueeze(-3)

        delta = (traj.unsqueeze(-2) - path).unsqueeze(-1)
        mahalanobis_square = (delta.mT @ inv_cov @ delta).squeeze()
        dist_min, _ = mahalanobis_square.min(dim=-1)

        # avg along trajectory
        costs = dist_min.sum(dim=-1) / dist_min.shape[-1]
        costs /= costs.max()
        return costs

    def distance_to_reference_path_final(
        self, trajectories, ego_shape, reference_path, desired_velocity, obstacle_predictions, lane_predictions
    ):
        """Calculate the mahalanobis distance to the reference path
        based on own state uncertainty
        Reference path is in vehicle coordinates
        """
        path = reference_path.coords
        traj = trajectories.coords[..., -1, :]
        inv_cov = torch.inverse(trajectories.coords_cov[:,-1,...]).unsqueeze(-3)

        delta = (traj.unsqueeze(-2) - path).unsqueeze(-1)
        mahalanobis_square = (delta.mT @ inv_cov @ delta).squeeze()
        dist_min, _ = mahalanobis_square.min(dim=-1)
        costs = dist_min
        costs /= costs.max()
        return costs

    def velocity_offset_final(
        self, trajectories, ego_shape, reference_path, desired_velocity, obstacle_predictions, lane_predictions
    ):
        """Calculate the velocity offset cost at end of trajectory"""
        try:
            # at last position
            costs = ((trajectories.v[..., -1] - desired_velocity) / desired_velocity).pow(2)
        except ValueError:
            raise NotImplementedError
        return costs

    def orientation_offset(
        self, trajectories, ego_shape, reference_path, desired_velocity, obstacle_predictions, lane_predictions
    ):
        """Calculate the offset to the orientation of the reference path along complete trajectory
        """
        path_coords = reference_path.coords
        traj_coords = trajectories.coords
        dist = torch.sqrt(torch.sum((traj_coords.unsqueeze(-2) - path_coords) ** 2, dim=-1))
        mask = dist.argmin(dim=-1, keepdim=False)
        # with mahalanobis distance:
        path_psi = reference_path.psi[mask]
        traj_psi = trajectories.psi
        inv_cov = 1/trajectories.covs[..., trajectories.psi_idx, trajectories.psi_idx]
        delta = path_psi - traj_psi
        mahalanobis_square = delta * inv_cov * delta
        # average along trajectory
        costs = mahalanobis_square.sum(dim=-1) / mahalanobis_square.shape[-1]
        costs /= costs.max()
        return costs

    def orientation_offset_final(
        self, trajectories, ego_shape, reference_path, desired_velocity, obstacle_predictions, lane_predictions
    ):
        """Calculate the offset to the orientation of the reference path only at last position
        Reference path is in vehicle coordinates -> distance only in lateral direction "y_loc"
        """
        path_coords = reference_path.coords
        traj_coords = trajectories.coords[..., -1, :]
        dist = torch.sqrt(torch.sum((traj_coords.unsqueeze(-2) - path_coords) ** 2, dim=-1))
        mask = dist.argmin(dim=-1, keepdim=False)
        # # with mahalanobis distance:
        path_psi = reference_path.psi[mask]
        traj_psi = trajectories.psi[..., -1]
        inv_cov = 1/trajectories.covs[..., -1, trajectories.psi_idx, trajectories.psi_idx]
        delta = path_psi - traj_psi

        mahalanobis_square = delta * inv_cov * delta

        costs = mahalanobis_square

        costs /= costs.max()
        return costs

    def prediction(self, trajectories, ego_shape, reference_path, desired_velocity, obstacle_predictions, lane_predictions):
        """Calculate the prediction costs based on the inverse mahalanobis distance"""
        inv_dists = torch.zeros([trajectories.num_batches, trajectories.num_states], dtype=DTYPE, device=self.device)
        trajs = trajectories.coords
        for pred in obstacle_predictions:
            states = pred["states"]
            pred_pos = states.coords
            inv_cov = torch.inverse(states.coords_cov)
            num_steps = min(pred_pos.shape[-2], trajs.shape[-2])
            delta = (trajs[..., :num_steps, :] - pred_pos[..., :num_steps, :]).unsqueeze(-1)
            mahalanobis_square = (delta.mT @ inv_cov[:num_steps, ...] @ delta).squeeze()
            inv_dists += 1 / mahalanobis_square
        # max along trajectory
        costs, _ = inv_dists.max(dim=-1)
        return costs

    def distance_to_boundary(
        self, trajectories, ego_shape, reference_path, desired_velocity, obstacle_predictions, lane_predictions
    ):
        """Calculate the cost based on the distance to the boundary"""
        # mahalanobis distance
        traj = trajectories.coords
        inv_dists = torch.zeros([trajectories.num_batches, trajectories.num_states], dtype=DTYPE, device=self.device)
        for pred in lane_predictions:
            if pred["boundary"] == True:
                boundary = pred["states"]
                coords = boundary.coords
                inv_cov = torch.inverse(boundary.coords_cov)
                delta = (traj.unsqueeze(-2) - coords).unsqueeze(-1)
                mahalanobis_square = (delta.mT @ inv_cov @ delta).squeeze()
                dist_min, _ = mahalanobis_square.min(dim=-1)
                dist_min = dist_min.clip(min=0.01)
                inv_dists += 1 / dist_min
        # max along trajectory
        costs, _ = inv_dists.max(dim=-1)
        return costs

