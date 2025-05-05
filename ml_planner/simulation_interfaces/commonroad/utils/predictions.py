__author__ = "Marc Kaufed"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

import logging
from typing import Dict, Union
from dataclasses import dataclass
import torch

import numpy as np
from commonroad.scenario.obstacle import ObstacleRole
from commonroad.scenario.scenario import Scenario
from commonroad_route_planner.route_generation_strategies.default_generation_strategy import DefaultGenerationStrategy
from prediction.main import WaleNet

from ml_planner.simulation_interfaces.commonroad.utils.sensor_model import get_visible_objects, get_obstacles_in_radius
from ml_planner.general_utils.data_types import StateTensor, DTYPE

# get loggers
msg_logger = logging.getLogger("Interface_Logger")


@dataclass
class CurrentPredictions:
    predictions: Dict[int, Dict[str, Union[np.ndarray, np.ndarray]]]

    def prediction_dict(self):
        return self.predictions

    def predictions_of_vehicle(self, vehicle_id: int):
        return self.predictions[vehicle_id]

    def create_StateTensors(self, device):
        predictions = []
        for pred_id, pred in self.predictions.items():

            states = StateTensor(
                {
                    "x": pred["pos_list"][:, 0],
                    "y": pred["pos_list"][:, 1],
                    "psi": pred["orientation_list"],
                    "v": pred["v_list"],
                },
                device=device,
            )

            covs = torch.zeros(
                [states.num_states, states.num_variables, states.num_variables], dtype=DTYPE, device=device
            )
            cov_tensor = torch.tensor(pred["cov_list"], dtype=DTYPE, device=device)
            num_pads = states.num_variables - cov_tensor.shape[-1]
            covs[..., : cov_tensor.shape[-1], : cov_tensor.shape[-1]] = cov_tensor
            covs[..., cov_tensor.shape[-1] :, cov_tensor.shape[-1] :] = torch.diag(0.01 * torch.ones(num_pads))
            states.add_cov(covs)
            predictions.append(
                {
                    "id": pred_id,
                    "states": states,
                    "shape": pred["shape"],
                    "types": {pred["type"]: 1},
                }
            )

        return predictions


class Predictor:
    def __init__(self, config, scenario: Scenario, planning_horizon: float):
        """Calculates the predictions for all obstacles in the scenario.

        :param config: The configuration.
        :param scenario: The scenario for which to calculate the predictions.
        :param planning_horizon: Time horizon of trajectories
        """
        self.config = config
        if config.mode == "walenet":
            self.predictor = WaleNet(scenario=scenario)
        elif config.mode == "semantic_cv":
            pass
        else:
            self.predictor = None
        self.scenario = scenario
        self.planning_horizon = planning_horizon

    def get_predictions(self, ego_state, timestep: int):
        """Calculate the predictions for all obstacles in the scenario.

        :param current_timestep: The timestep after which to start predicting.
        :param ego_state: current state of the ego vehicle.
        """
        predictions = None
        obstacle_list, vis_area = self.get_visible_objects(ego_state, timestep)
        if self.config.mode == "walenet":
            # Calculate predictions for all obstacles using WaleNet.
            predictions = self.walenet_prediction(obstacle_list, timestep)
        elif self.config.mode == "semantic_cv":
            # Calculate predictions for all obstacles using semantic cv.
            predictions = self.semantic_cv_prediction(obstacle_list, timestep)
        else:
            # ground_truth
            predictions = self.get_ground_truth_prediction(obstacle_list, timestep)

        return CurrentPredictions(predictions), vis_area

    def get_visible_objects(self, ego_state, timestep: int):
        """Calculate the visible obstacles for the ego vehicle."""
        if self.config.cone_angle > 0:
            vehicles_in_cone_angle = True
        else:
            vehicles_in_cone_angle = False
        if self.config.calc_visible_area:
            visible_obstacles, visible_area = get_visible_objects(
                scenario=self.scenario,
                time_step=timestep,
                ego_state=ego_state,
                sensor_radius=self.config.sensor_radius,
                ego_id=ego_state.obstacle_id,
                vehicles_in_cone_angle=vehicles_in_cone_angle,
                config=self.config,
            )
            return visible_obstacles, visible_area
        else:
            visible_obstacles = get_obstacles_in_radius(
                scenario=self.scenario,
                ego_id=ego_state.obstacle_id,
                ego_state=ego_state,
                time_step=timestep,
                radius=self.config.sensor_radius,
                vehicles_in_cone_angle=vehicles_in_cone_angle,
                config=self.config,
            )
            return visible_obstacles, None

    def walenet_prediction(self, visible_obstacles, timestep):
        """Calculate the predictions for all obstacles in the scenario using WaleNet."""
        (dyn_visible_obstacles, stat_visible_obstacles) = get_dyn_and_stat_obstacles(
            scenario=self.scenario, obstacle_ids=visible_obstacles
        )

        # get prediction for dynamic obstacles
        predictions = self.predictor.step(
            time_step=timestep, obstacle_id_list=dyn_visible_obstacles, scenario=self.scenario
        )
        pred_horizon = int(self.planning_horizon / self.scenario.dt) + 1
        for key in predictions.keys():
            predictions[key]["pos_list"] = predictions[key]["pos_list"][:pred_horizon]
            predictions[key]["cov_list"] = predictions[key]["cov_list"][:pred_horizon]
        # create and add prediction of static obstacles
        predictions = add_static_obstacle_to_prediction(
            scenario=self.scenario,
            predictions=predictions,
            obstacle_id_list=stat_visible_obstacles,
            pred_horizon=int(self.planning_horizon / self.scenario.dt),
        )
        predictions = get_orientation_velocity_and_shape_of_prediction(predictions=predictions, scenario=self.scenario)

        return predictions

    def get_ground_truth_prediction(self, visible_obstacles, timestep: int):
        """
        Transform the ground truth to a prediction. Use this if the prediction fails.

        Args:
            obstacle_ids ([int]): IDs of the visible obstacles.
            scenario (Scenario): considered scenario.
            time_step (int): Current time step.
            pred_horizon (int): Prediction horizon for the prediction.

        Returns:
            dict: Dictionary with the predictions.
        """
        # get the prediction horizon
        pred_horizon = int(self.planning_horizon / self.scenario.dt) + 1
        # create a dictionary for the predictions
        prediction_result = {}
        for obstacle_id in visible_obstacles:
            try:
                obstacle = self.scenario.obstacle_by_id(obstacle_id)
                fut_pos = []
                fut_cov = []
                fut_yaw = []
                fut_v = []
                # predict dynamic obstacles as long as they are in the scenario
                if obstacle.obstacle_role == ObstacleRole.DYNAMIC:
                    len_pred = len(obstacle.prediction.occupancy_set)
                # predict static obstacles for the length of the prediction horizon
                else:
                    len_pred = pred_horizon
                # create mean and the covariance matrix of the obstacles
                for ts in range(timestep, min(pred_horizon + timestep, len_pred)):
                    occupancy = obstacle.occupancy_at_time(ts)
                    if occupancy is not None:
                        # create mean and covariance matrix
                        fut_pos.append(occupancy.shape.center)
                        fut_cov.append([[1.0, 0.0], [0.0, 1.0]])
                        fut_yaw.append(occupancy.shape.orientation)
                        fut_v.append(obstacle.prediction.trajectory.state_list[ts].velocity)

                fut_pos = np.array(fut_pos)
                fut_cov = np.array(fut_cov)
                fut_yaw = np.array(fut_yaw)
                fut_v = np.array(fut_v)

                shape_obs = {"length": obstacle.obstacle_shape.length, "width": obstacle.obstacle_shape.width}
                # add the prediction for the considered obstacle
                prediction_result[obstacle_id] = {
                    "pos_list": fut_pos,
                    "cov_list": fut_cov,
                    "orientation_list": fut_yaw,
                    "v_list": fut_v,
                    "shape": shape_obs,
                    "type": obstacle.obstacle_type.value,
                }
            except Exception as e:
                msg_logger.warning(f"Could not calculate ground truth prediction for obstacle {obstacle_id}: ", e)

        return prediction_result

    def semantic_cv_prediction(self, visible_obstacles, timestep: int):
        """Calculate the predictions for all obstacles in the scenario using semantic cv."""
        # get the prediction horizon
        num_steps = int(self.planning_horizon / self.scenario.dt) + 1
        # create a dictionary for the predictions
        prediction_result = {}
        for obstacle_id in visible_obstacles:
            try:
                obstacle = self.scenario.obstacle_by_id(obstacle_id)
                obs_state = obstacle.state_at_time(timestep)
                ini_pos = obs_state.position
                # get traveled distance at current velocity
                v = obs_state.velocity
                dist = v * self.planning_horizon
                # get the lanelet of the obstacle
                start_lanelet_ids = self.scenario.lanelet_network.find_lanelet_by_position([ini_pos])[0]
                all_routes = []
                for start_id in start_lanelet_ids:
                    # Calculate all routes using recursive function
                    routes = find_all_routes(self.scenario.lanelet_network, start_id, max_depth=33)
                    all_routes.extend(routes)
                # Convert "graph ids" to real route using commonroad Route class
                ref_path_candidates = [
                    DefaultGenerationStrategy.generate_route(
                        lanelet_network=self.scenario.lanelet_network,
                        lanelet_ids=route,
                        goal_region=None,
                        initial_state=None,
                    )
                    for route in all_routes
                ]
                fut_pos = np.array([])
                fut_cov = np.array([])
                fut_yaw = np.array([])
                fut_v = np.array([])
                for idx, ref_path in enumerate(ref_path_candidates):
                    path = ref_path.get_route_slice_from_position(
                        ini_pos[0], ini_pos[1], distance_ahead_in_m=dist, distance_behind_in_m=0.0
                    )
                    delta = ini_pos - path.reference_path[0]
                    fut_pos = path.reference_path + delta
                    fut_yaw = path.path_orientation
                    original_indices = np.linspace(0, len(fut_pos) - 1, len(fut_pos))
                    resampled_indices = np.linspace(0, len(fut_pos) - 1, num_steps)
                    fut_pos = np.stack(
                        [
                            np.interp(resampled_indices, original_indices, fut_pos[:, 0]),
                            np.interp(resampled_indices, original_indices, fut_pos[:, 1]),
                        ],
                        axis=-1,
                    )
                    fut_yaw = np.interp(resampled_indices, original_indices, fut_yaw)
                    fut_v = np.repeat(v, num_steps)

                    # add noice (based on /github.com/balzer82/Kalman/blob/master/Kalman-Filter-CV)
                    # uncertainties in x,y, vx, vy
                    P = [
                        np.diag(
                            [
                                self.config.uncertainties.x,
                                self.config.uncertainties.y,
                                self.config.uncertainties.vx,
                                self.config.uncertainties.vy,
                            ]
                        )
                    ]
                    # state transition matrix
                    A = np.array([[1, 0, self.scenario.dt, 0], [0, 1, 0, self.scenario.dt], [0, 0, 1, 0], [0, 0, 0, 1]])
                    # process noise covariance in object coordinates
                    sv = self.config.uncertainties.process_noice
                    self.Q = (
                        np.array(
                            [
                                [(self.scenario.dt**4) / 4, 0, (self.scenario.dt**3) / 2, 0],
                                [0, (self.scenario.dt**4) / 4, 0, (self.scenario.dt**3) / 2],
                                [(self.scenario.dt**3) / 2, 0, self.scenario.dt**2, 0],
                                [0, (self.scenario.dt**3) / 2, 0, self.scenario.dt**2],
                            ]
                        )
                        * sv**2
                    )
                    for i in range(num_steps - 1):
                        P.append(A @ P[-1] @ A.T + self.Q)
                    P = np.array(P)
                    cov = P[..., :2, :2]
                    rot = np.array([[np.cos(fut_yaw), np.sin(fut_yaw)], [-np.sin(fut_yaw), np.cos(fut_yaw)]]).T
                    clip_cov = True
                    i = -1
                    while clip_cov:

                        loc_pos = fut_pos[i]
                        loc_cov = cov[i]
                        lanelet_id = self.scenario.lanelet_network.find_lanelet_by_position([loc_pos])[0]
                        lanelet = self.scenario.lanelet_network.find_lanelet_by_id(lanelet_id[0])

                        center_idx = np.argmin(np.sum((lanelet.center_vertices - loc_pos) ** 2, axis=1))
                        (_, right, left, _) = lanelet.interpolate_position(lanelet.distance[center_idx])

                        r_max = max(np.linalg.norm(right - loc_pos), np.linalg.norm(left - loc_pos)) + 0.5
                        if r_max < loc_cov[-1, -1]:
                            cov[i, -1, -1] = r_max
                            i -= 1
                        else:
                            clip_cov = False
                    fut_cov = rot @ cov @ np.swapaxes(rot, -1, -2)
                    shape_obs = {"length": obstacle.obstacle_shape.length, "width": obstacle.obstacle_shape.width}
                    # add the prediction for the considered obstacle

                    prediction_result[str(obstacle_id) + f"_{idx}"] = {
                        "pos_list": fut_pos,
                        "cov_list": fut_cov,
                        "orientation_list": fut_yaw,
                        "v_list": fut_v,
                        "shape": shape_obs,
                        "type": obstacle.obstacle_type.value,
                    }
            except Exception as e:
                msg_logger.warning(f"Could not calculate semantic cv prediction for obstacle {obstacle_id}: ", e)

        return prediction_result


def get_dyn_and_stat_obstacles(obstacle_ids, scenario):
    """
    Split a set of obstacles in a set of dynamic obstacles and a set of static obstacles.

    Args:
        obstacle_ids ([int]): IDs of all considered obstacles.
        scenario: Considered scenario.

    Returns:
        [int]: List with the IDs of all dynamic obstacles.
        [int]: List with the IDs of all static obstacles.

    """
    dyn_obstacles = []
    stat_obstacles = []
    for obst_id in obstacle_ids:
        if scenario.obstacle_by_id(obst_id).obstacle_role == ObstacleRole.DYNAMIC:
            dyn_obstacles.append(obst_id)
        else:
            stat_obstacles.append(obst_id)

    return dyn_obstacles, stat_obstacles


def get_orientation_velocity_and_shape_of_prediction(
    predictions: dict, scenario, safety_margin_length=0.5, safety_margin_width=0.2
):
    """
    Extend the prediction by adding information about the orientation, velocity and the shape of the predicted obstacle.

    Args:
        predictions (dict): Prediction dictionary that should be extended.
        scenario (Scenario): Considered scenario.

    Returns:
        dict: Extended prediction dictionary.
    """
    # go through every predicted obstacle
    obstacle_ids = list(predictions.keys())
    for obstacle_id in obstacle_ids:
        obstacle = scenario.obstacle_by_id(obstacle_id)
        # get x- and y-position of the predicted trajectory
        pred_traj = predictions[obstacle_id]["pos_list"]
        pred_length = len(pred_traj)

        # there may be some predictions without any trajectory (when the obstacle disappears due to exceeding time)
        if pred_length == 0:
            del predictions[obstacle_id]
            continue

        # for predictions with only one timestep, the gradient can not be derived --> use initial orientation
        if pred_length == 1:
            pred_orientation = [obstacle.initial_state.orientation]
            pred_v = [obstacle.initial_state.velocity]
        else:
            t = [0.0 + i * scenario.dt for i in range(pred_length)]
            x = pred_traj[:, 0][0:pred_length]
            y = pred_traj[:, 1][0:pred_length]

            # calculate the yaw angle for the predicted trajectory
            dx = np.gradient(x, t)
            dy = np.gradient(y, t)
            # if the vehicle does barely move, use the initial orientation
            # otherwise small uncertainties in the position can lead to great orientation uncertainties
            if all(dxi < 0.0001 for dxi in dx) and all(dyi < 0.0001 for dyi in dy):
                init_orientation = obstacle.initial_state.orientation
                pred_orientation = np.full((1, pred_length), init_orientation)[0]
            # if the vehicle moves, calculate the orientation
            else:
                pred_orientation = np.arctan2(dy, dx)

            # get the velocity from the derivation of the position
            pred_v = np.sqrt((np.power(dx, 2) + np.power(dy, 2)))

        # add the new information to the prediction dictionary
        predictions[obstacle_id]["orientation_list"] = pred_orientation
        predictions[obstacle_id]["v_list"] = pred_v
        obstacle_shape = obstacle.obstacle_shape
        predictions[obstacle_id]["shape"] = {
            "length": obstacle_shape.length + safety_margin_length,
            "width": obstacle_shape.width + safety_margin_width,
        }
        predictions[obstacle_id]["type"] = obstacle.obstacle_type.value

    # return the updated predictions dictionary
    return predictions


def add_static_obstacle_to_prediction(predictions: dict, obstacle_id_list, scenario, pred_horizon: int = 50):
    """
    Add static obstacles to the prediction since predictor can not handle static obstacles.

    Args:
        predictions (dict): Dictionary with the predictions.
        obstacle_id_list ([int]): List with the IDs of the static obstacles.
        scenario (Scenario): Considered scenario.
        pred_horizon (int): Considered prediction horizon. Defaults to 50.

    Returns:
        dict: Dictionary with the predictions.
    """
    for obstacle_id in obstacle_id_list:
        obstacle = scenario.obstacle_by_id(obstacle_id)
        fut_pos = []
        fut_cov = []
        # create a mean and covariance matrix for every time step in the prediction horizon
        for ts in range(int(pred_horizon)):
            fut_pos.append(obstacle.initial_state.position)
            fut_cov.append([[0.02, 0.0], [0.0, 0.02]])

        fut_pos = np.array(fut_pos)
        fut_cov = np.array(fut_cov)
        raise ValueError
        # TODO Marc: shape of static obstacle etc. added?
        # add the prediction to the prediction dictionary
        predictions[obstacle_id] = {"pos_list": fut_pos, "cov_list": fut_cov}

    return predictions


def find_all_routes(lanelet_network, start_lanelet_id, max_depth=2):
    routes = []
    explore_routes(lanelet_network, start_lanelet_id, [], routes, 0, max_depth)
    return routes


def explore_routes(lanelet_network, current_lanelet_id, route, all_routes, depth, max_depth):
    # Add current lanelet to the route
    route.append(current_lanelet_id)

    lanelet = lanelet_network.find_lanelet_by_id(current_lanelet_id)
    successors = []
    if lanelet.successor:
        successors.extend(lanelet.successor)
    if lanelet.adj_right and lanelet.adj_right_same_direction:
        lanelet_adj_right = lanelet_network.find_lanelet_by_id(lanelet.adj_right)
        if lanelet_adj_right.successor:
            successors.append(lanelet.adj_right)
    if lanelet.adj_left and lanelet.adj_left_same_direction:
        lanelet_adj_left = lanelet_network.find_lanelet_by_id(lanelet.adj_left)
        if lanelet_adj_left.successor:
            successors.append(lanelet.adj_left)

    if depth >= max_depth:
        successors = []
        # Max depth reached, return without exploring further

    if not successors:
        # If no successors, save route and return
        all_routes.append(route.copy())
        return

    for successor in successors:
        explore_routes(lanelet_network, successor, route, all_routes, depth + 1, max_depth)
        route.pop()  # Backtrack to explore other paths
