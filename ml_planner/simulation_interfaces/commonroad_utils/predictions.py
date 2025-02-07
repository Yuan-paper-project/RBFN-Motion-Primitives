__author__ = "Marc Kaufed"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

import logging
from typing import Dict, Union
from dataclasses import dataclass

import numpy as np
from commonroad.scenario.obstacle import ObstacleRole
from commonroad.scenario.scenario import Scenario


from prediction.main import WaleNet
from ml_planner.simulation_interfaces.commonroad_utils.sensor_model import get_visible_objects, get_obstacles_in_radius

# get logger
msg_logger = logging.getLogger("Interface_Logger")


@dataclass
class CurrentPredictions():
    predictions: Dict[int, Dict[str, Union[np.ndarray, np.ndarray]]]

    def prediction_dict(self):
        return self.predictions

    def get_predictions_of_vehicle(self, vehicle_id: int):
        return self.predictions[vehicle_id]


class Predictor():
    def __init__(self, config, scenario: Scenario, planning_horizon: float):
        """ Calculates the predictions for all obstacles in the scenario.

        :param config: The configuration.
        :param scenario: The scenario for which to calculate the predictions.
        :param planning_horizon: Time horizon of trajectories
        """
        self.config = config
        self.predictor = WaleNet(scenario=scenario) if config.mode == "walenet" else None
        self.scenario = scenario
        self.planning_horizon = planning_horizon

    def get_predictions(self, ego_state, timestep: int):
        """ Calculate the predictions for all obstacles in the scenario.

        :param current_timestep: The timestep after which to start predicting.
        :param ego_state: current state of the ego vehicle.
        """
        predictions = None
        obstacle_list, vis_area = self.get_visible_objects(ego_state, timestep)
        if self.predictor:
            # Calculate predictions for all obstacles using WaleNet.
            predictions = self.walenet_prediction(obstacle_list, timestep)
        else:
            # ground_truth
            predictions = self.get_ground_truth_prediction(obstacle_list,  timestep)

        return CurrentPredictions(predictions)

    def get_visible_objects(self, ego_state, timestep: int):
        """Calculate the visible obstacles for the ego vehicle."""
        if self.config.cone_angle > 0:
            vehicles_in_cone_angle = True
        else:
            vehicles_in_cone_angle = False
        if self.config.calc_visible_area:
            try:
                visible_obstacles, visible_area = get_visible_objects(
                        scenario=self.scenario,
                        time_step=timestep,
                        ego_state=ego_state,
                        sensor_radius=self.config.sensor_radius,
                        ego_id=ego_state.obstacle_id,
                        vehicles_in_cone_angle=vehicles_in_cone_angle,
                        config=self.config
                    )
                return visible_obstacles, visible_area
            except:
                msg_logger.warning("Could not calculate visible area!")
                visible_obstacles = get_obstacles_in_radius(
                    scenario=self.scenario,
                    ego_id=ego_state.obstacle_id,
                    ego_state=ego_state,
                    time_step=timestep,
                    radius=self.config.sensor_radius,
                    vehicles_in_cone_angle=vehicles_in_cone_angle,
                    config=self.config
                )
                return visible_obstacles, None
        else:
            visible_obstacles = get_obstacles_in_radius(
                scenario=self.scenario,
                ego_id=ego_state.obstacle_id,
                ego_state=ego_state,
                time_step=timestep,
                radius=self.config.sensor_radius,
                vehicles_in_cone_angle=vehicles_in_cone_angle,
                config=self.config
            )
            return visible_obstacles, None

    def walenet_prediction(self, visible_obstacles, timestep):
        """Calculate the predictions for all obstacles in the scenario using WaleNet."""
        (dyn_visible_obstacles,
         stat_visible_obstacles) = get_dyn_and_stat_obstacles(scenario=self.scenario,
                                                              obstacle_ids=visible_obstacles)

        # get prediction for dynamic obstacles
        predictions = self.predictor.step(time_step=timestep,
                                          obstacle_id_list=dyn_visible_obstacles,
                                          scenario=self.scenario)
        pred_horizon = int(self.planning_horizon / self.scenario.dt) + 1
        for key in predictions.keys():
            predictions[key]['pos_list'] = predictions[key]['pos_list'][:pred_horizon]
            predictions[key]['cov_list'] = predictions[key]['cov_list'][:pred_horizon]
        # create and add prediction of static obstacles
        predictions = add_static_obstacle_to_prediction(scenario=self.scenario,
                                                        predictions=predictions,
                                                        obstacle_id_list=stat_visible_obstacles,
                                                        pred_horizon=int(self.planning_horizon / self.scenario.dt))
        predictions = get_orientation_velocity_and_shape_of_prediction(predictions=predictions,
                                                                       scenario=self.scenario)

        return predictions

    def get_ground_truth_prediction(self, visible_obstacles, time_step: int):
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
                for ts in range(time_step, min(pred_horizon + time_step, len_pred)):
                    occupancy = obstacle.occupancy_at_time(ts)
                    if occupancy is not None:
                        # create mean and covariance matrix
                        fut_pos.append(occupancy.shape.center)
                        fut_cov.append([[1., 0.0], [0.0, 1.]])
                        fut_yaw.append(occupancy.shape.orientation)
                        fut_v.append(obstacle.prediction.trajectory.state_list[ts].velocity)

                fut_pos = np.array(fut_pos)
                fut_cov = np.array(fut_cov)
                fut_yaw = np.array(fut_yaw)
                fut_v = np.array(fut_v)

                shape_obs = {'length': obstacle.obstacle_shape.length, 'width': obstacle.obstacle_shape.width}
                # add the prediction for the considered obstacle
                prediction_result[obstacle_id] = {'pos_list': fut_pos, 'cov_list': fut_cov, 'orientation_list': fut_yaw,
                                                  'v_list': fut_v, 'shape': shape_obs}
            except Exception as e:
                msg_logger.warning(f"Could not calculate ground truth prediction for obstacle {obstacle_id}: ", e)

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
        pred_traj = predictions[obstacle_id]['pos_list']
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
        predictions[obstacle_id]['orientation_list'] = pred_orientation
        predictions[obstacle_id]['v_list'] = pred_v
        obstacle_shape = obstacle.obstacle_shape
        predictions[obstacle_id]['shape'] = {
            'length': obstacle_shape.length + safety_margin_length,
            'width': obstacle_shape.width + safety_margin_width,
        }

    # return the updated predictions dictionary
    return predictions


def add_static_obstacle_to_prediction(
        predictions: dict, obstacle_id_list, scenario, pred_horizon: int = 50
):
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

        # add the prediction to the prediction dictionary
        predictions[obstacle_id] = {'pos_list': fut_pos, 'cov_list': fut_cov}

    return predictions
