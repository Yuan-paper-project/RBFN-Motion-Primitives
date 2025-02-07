__author__ = "Marc Kaufed"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

from abc import ABC
from pathlib import Path
from shapely import Point, box
import numpy as np
import torch

from vehiclemodels.vehicle_parameters import setup_vehicle_parameters

from ml_planner.simulation_interfaces.commonroad_utils.predictions import CurrentPredictions
from ml_planner.helper_functions.logging import logger_initialization
from ml_planner.helper_functions.data_types import StateVector, DynamicModel, VehicleModel
from ml_planner.planner.planner_utils import (load_model, convert_globals_to_locals,
                                              rotate_covariance_matrix, convert_locals_to_globals)
from ml_planner.planner.networks.utils import DTYPE
from ml_planner.analytic_solution.utils import get_vehicle_dynamics_system
from ml_planner.analytic_solution.acceleration_constraints import acceleration_constraints
from ml_planner.planner.cost_functions import Costs


class MLSamplingPlanner(ABC):
    def __init__(self, model_name, model_type, dynamic_model, vehicle_model, log_path, sample_feasible_region,
                 latest_resampling_every_n_steps, use_improved_trajectories,
                 cost_weights, dx, y_min, y_max, dy, psi_min, psi_max, dpsi, log_level="warning"):

        self.msg_logger = logger_initialization(log_path=log_path, logger="ML_Planner",
                                                loglevel=log_level.upper())
        cur_dir = Path(__file__).parent
        model_path = cur_dir / "models" / model_name
        # get device
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.msg_logger.info(f"Using {self.device} device")
        # load and initialize model
        model, self.info, self.dT, self.horizon = load_model(model_path, model_type, self.device)
        self.model = model.to(self.device)
        self.input_labels = self.model.input_labels
        self.output_labels = self.model.output_labels
        self.initial_labels = [label for label in self.input_labels if '0' in label]
        self.idx_initial_labels = None
        self.msg_logger.info(self.model)

        # set configurations
        self.sample_feasible_region = sample_feasible_region
        self.resample_every_n_steps = latest_resampling_every_n_steps
        self.use_improved_trajectories = use_improved_trajectories
        self.resampling_counter = -1

        # initialize variables
        self.reference_path = None
        self.current_state: StateVector = None
        self.current_predictions: CurrentPredictions = None
        self.current_road_boundary = None
        self.current_desired_velocity = None
        self.cost_function = Costs(cost_weights, self.output_labels, self.device)
        self.trajectories_local = None
        self.optimal_trajectory = None
        self.total_costs = np.inf
        self.costs = None
        self.x0_loc = None
        # sampling grid
        # x range calcualted based on current velocity
        self.dx = max(dx, self.info['dx'])
        # y range
        self.dy = max(dy, self.info['dy'])
        num_y = int((y_max-y_min)/self.dy)+1
        self.y_range = torch.linspace(y_min, y_max, num_y, device=self.device, dtype=DTYPE)
        # psi range
        self.dpsi = max(dpsi, self.info['dpsi'])
        num_psi = int((psi_max-psi_min)/self.dpsi) + 1
        self.psi_range = torch.linspace(psi_min, psi_max, num_psi, device=self.device, dtype=DTYPE)

        # get vehicle parameters
        dynamic_model = DynamicModel[dynamic_model]
        vehicle_model = VehicleModel[vehicle_model]
        self.vehicle_params = setup_vehicle_parameters(vehicle_model.value)
        _, _, p = get_vehicle_dynamics_system(dynamic_model, self.vehicle_params)
        self.acc_limits = p[2]

    def update_externals(self, reference_path=None,
                         current_state: StateVector = None,
                         current_predictions: CurrentPredictions = None,
                         current_road_boundary=None,
                         current_desired_velocity=None,
                         ):
        """get new data from external sources"""
        if reference_path is not None:
            self.reference_path = torch.tensor(reference_path, dtype=DTYPE, device=self.device)
        if current_state is not None:
            self.current_state = current_state
        if current_predictions is not None:
            self.current_predictions = current_predictions
        if current_road_boundary is not None:
            self.current_road_boundary = current_road_boundary
        if current_desired_velocity is not None:
            self.current_desired_velocity = current_desired_velocity

    @property
    def all_trajectories(self):
        all_trajectories = []
        for idx in range(self.trajectories_local.shape[0]):
            all_trajectories.append(self._convert_to_global(self.trajectories_local[idx]))
        return all_trajectories

    def plan(self):
        """calculate optimal trajectory for current state"""
        self.resampling_counter += 1
        self.msg_logger.debug("Calculate Trajectory")
        self.model.eval()
        with torch.no_grad():
            # sample trajectories in local reference frame
            self.trajectories_local = self._sample_trajectories()
            # calculate costs
            total_costs, costs = self._get_trajectory_costs(self.trajectories_local)

            min_costs, idx_min_cost = torch.min(total_costs, dim=0, keepdim=True)

        resampling = self.resampling_counter % self.resample_every_n_steps == 0
        improved = min_costs < self.total_costs and self.use_improved_trajectories
        if resampling or improved:
            # if resampling or improved update optimal trajectory
            self.msg_logger.debug("Resample trajectory" if resampling else "Found improved trajectory")
            self.total_costs = min_costs
            self.costs = costs[idx_min_cost]
            optimal_traj_local = self.trajectories_local[idx_min_cost, :, :].squeeze()

            # transform to global reference frame
            self.optimal_trajectory = self._convert_to_global(optimal_traj_local)

            self.msg_logger.debug(f"Optimal Trajectory with costs {self.total_costs.cpu().detach().numpy()} ")
            self.msg_logger.debug(f"Cost weights: {self.cost_function.cost_terms} ")
            self.msg_logger.debug(f"wo weight: {self.costs.cpu().detach().numpy()} ")
            self.msg_logger.debug(f"""w weights: {self.costs.cpu().detach().numpy() *
                                  self.cost_function.cost_weights.cpu().detach().numpy()}""")

        else:
            self.msg_logger.debug("Using previously calculated optimal trajectory")
            self.optimal_trajectory = {key: value[1:] for key, value in self.optimal_trajectory.items()}
            pass
        return self.optimal_trajectory

    def _sample_trajectories(self):
        """sample trajectories in local reference frame"""
        # initial state:
        self.x0_loc = torch.zeros(len(self.initial_labels), dtype=DTYPE, device=self.device)
        self.x0_loc[self.initial_labels.index('v_0')] = np.round(float(self.current_state.v))
        # final states:
        a_pos = acceleration_constraints(self.current_state.v, self.acc_limits[0], self.acc_limits)
        a_neg = acceleration_constraints(self.current_state.v, -self.acc_limits[0], self.acc_limits)
        x_min = np.ceil(self.current_state.v**2/(-2*a_neg))
        x_max = self.current_state.v*self.horizon + 0.5*a_pos*self.horizon**2
        num_x = int((x_max-x_min)/self.dx)+1
        x_range = torch.linspace(x_min, x_max, num_x, device=self.device, dtype=DTYPE)
        xy_loc = torch.cartesian_prod(x_range, self.y_range)

        if self.sample_feasible_region:
            # sample within driveable region, avoid obstacles etc.

            xy_glob = convert_locals_to_globals(xy_loc, self.current_state)

            # get predicted positions of obstacles at end of planning horizon
            fut_obs = []
            if self.current_predictions.predictions:
                fut_obs = [box(xmin=pred['pos_list'][-1][0] - pred["shape"]["length"]/2,
                               xmax=pred['pos_list'][-1][0] + pred["shape"]["length"]/2,
                               ymin=pred['pos_list'][-1][1] - pred["shape"]["length"]/2,
                               ymax=pred['pos_list'][-1][1] + pred["shape"]["length"]/2)
                           for pred in self.current_predictions.predictions.values() if len(pred['pos_list'])]

            # create a mask for all points that are within the road boundary and not within obstacles
            mask = [self.current_road_boundary.contains(Point(xy_glob[i].cpu()))
                    and not any([ob.contains(Point(xy_glob[i].cpu())) for ob in fut_obs])
                    for i in range(xy_glob.size(0))]
            # filter out points that are non-feasible
            xy_loc = xy_loc[mask]

        psi_range = self.psi_range.repeat(xy_loc.shape[0])
        xy_loc = xy_loc.repeat_interleave(self.psi_range.shape[0], dim=0)
        xF = torch.cat((xy_loc, psi_range.unsqueeze(1)), dim=1)

        # only sample trajectoris that do not come back to current "local centerline"
        mask = (xF[:, -2] * xF[:, -1]) >= 0
        xF = xF[mask]

        x0_loc = self.x0_loc.unsqueeze(0).expand(xF.shape[0], self.x0_loc.shape[0])
        input_x = torch.hstack([x0_loc, xF]).to(self.device)
        # optain trajectories
        pred = self.model(input_x)

        # remove invalid trajectories (e.g. trajectories that go backwards)
        mask = torch.all(pred[:, 0, 1:] >= pred[:, 0, :-1], dim=1)
        pred = pred[mask]

        self.msg_logger.debug(f"Sampled {pred.shape[0]} trajectories")
        return pred

    def _get_trajectory_costs(self, trajectories):
        # local ref path
        current_position = torch.tensor([self.current_state.x, self.current_state.y], dtype=DTYPE, device=self.device)
        distances = torch.sqrt(torch.sum((self.reference_path[:, :2] - current_position) ** 2, axis=1))
        mask = abs(distances) < 200
        part_ref_path = self.reference_path[mask]
        ref_path_local = convert_globals_to_locals(part_ref_path, self.current_state)
        ref_path_local = ref_path_local.squeeze()
        # local predictions
        predictions = None
        if self.current_predictions.predictions:
            pred_pos = torch.tensor(np.array([i['pos_list'] for i in self.current_predictions.predictions.values()]),
                                    dtype=DTYPE, device=self.device)
            pred_cov = torch.tensor(np.array([i['cov_list'] for i in self.current_predictions.predictions.values()]),
                                    dtype=DTYPE, device=self.device)
            if pred_pos.sum() != 0:
                pred_pos_local = convert_globals_to_locals(pred_pos, self.current_state)
                pred_pos_local = pred_pos_local.reshape(pred_pos.shape)
                pred_pos_local_left = pred_pos_local + torch.tensor([0, self.vehicle_params.w],
                                                                    dtype=DTYPE, device=self.device)
                pred_pos_local_right = pred_pos_local - torch.tensor([0, self.vehicle_params.w], dtype=DTYPE,
                                                                     device=self.device)
                pred_pos_local = torch.cat((pred_pos_local_left, pred_pos_local_right), dim=0)
                pred_cov_local = rotate_covariance_matrix(pred_cov, self.current_state)
                pred_cov_local = torch.cat((pred_cov_local, pred_cov_local), dim=0)
                predictions = [pred_pos_local, pred_cov_local]
        # local boundary
        x, y = self.current_road_boundary.exterior.xy
        boundary = torch.tensor([x, y], dtype=DTYPE, device=self.device).T
        boundary_local = convert_globals_to_locals(boundary, self.current_state)

        # calculate costs
        total_costs, costs = self.cost_function.calculate(trajectories,
                                                          ref_path_local,
                                                          self.current_desired_velocity,
                                                          predictions,
                                                          boundary_local)
        return total_costs, costs

    def _convert_to_global(self, trajectory):
        """ convert local trajectory from local to global reference frame"""
        x_idx = self.output_labels.index('x')
        y_idx = self.output_labels.index('y')
        psi_idx = self.output_labels.index('psi')
        path_local = trajectory[[x_idx, y_idx, psi_idx], :].T

        path_global = convert_locals_to_globals(path_local, self.current_state)

        global_trajectory = {'x': path_global[:, 0].cpu().detach().numpy(),
                             'y': path_global[:, 1].cpu().detach().numpy(),
                             'psi': path_global[:, 2].cpu().detach().numpy(),
                             'v': trajectory[self.output_labels.index('v'), :].cpu().detach().numpy(),
                             }
        return global_trajectory
