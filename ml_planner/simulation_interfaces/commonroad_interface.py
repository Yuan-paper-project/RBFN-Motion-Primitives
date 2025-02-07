__author__ = "Marc Kaufed"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

from abc import ABC
import os
from typing import cast
from hydra.utils import instantiate
import matplotlib
import matplotlib.pyplot as plt
import imageio.v3 as iio
from PIL import Image
import numpy as np
from shapely import LineString

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.draw_params import MPDrawParams, DynamicObstacleParams, ShapeParams
from commonroad.geometry.shape import Rectangle as CRRectangle
from commonroad_route_planner.route_planner import RoutePlanner
from commonroad_route_planner.reference_path_planner import ReferencePathPlanner
from commonroad_velocity_planner.velocity_planner_interface import IVelocityPlanner
from commonroad_velocity_planner.configuration.configuration_builder import ConfigurationBuilder
from commonroad_velocity_planner.velocity_planning_problem import VppBuilder
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.geometry.shape import Rectangle as ShapeRectangle
from commonroad.scenario.state import CustomState, InitialState
from commonroad.scenario.trajectory import Trajectory

from ml_planner.planner.ml_sampling_planner import MLSamplingPlanner
from ml_planner.helper_functions.data_types import StateVector
from ml_planner.helper_functions.logging import logger_initialization
from ml_planner.simulation_interfaces.commonroad_utils.predictions import Predictor
from ml_planner.simulation_interfaces.commonroad_utils.visualization_utils import draw_uncertain_predictions


class CommonroadInterface(ABC):
    """Interface between CommonRoad and MP-RBFN planner"""
    def __init__(self, planner,
                 prediction_config,
                 interface_logging_level,
                 planner_logging_level,
                 log_path,
                 max_steps,
                 visualization_config,
                 scenario_path):
        super().__init__()
        self.log_path = log_path
        self.msg_logger = logger_initialization(log_path=log_path, logger="Interface_Logger",
                                                loglevel=interface_logging_level.upper())
        # load scenario and planning problem
        self.scenario, self.planning_problem_set = CommonRoadFileReader(scenario_path).open()
        self.planning_problem = list(self.planning_problem_set.planning_problem_dict.values())[0]

        # initialize route planner
        self.route_planner = RoutePlanner(lanelet_network=self.scenario.lanelet_network,
                                          planning_problem=self.planning_problem,
                                          scenario=self.scenario,
                                          extended_search=False)

        # initialize velocity planner
        self.velocity_planner = IVelocityPlanner()

        # initialize MP-RBFN sampling planner
        self.trajectory_planner = cast(MLSamplingPlanner,
                                       instantiate(planner.planner,
                                                   log_path=log_path,
                                                   log_level=planner_logging_level))

        # initialize prediction module
        self.predictor = Predictor(prediction_config,
                                   self.scenario,
                                   self.trajectory_planner.horizon)

        # initialize variables
        self.global_trajectory = None
        self.road_boundary = None
        self.goal_reached = False
        self.current_state = None
        self.optimal_trajectory = None
        self.state_list = []
        self.cr_obstacle_list = []
        self.predictions = None
        self.timestep = 0
        self.max_time_steps_scenario = int(
                max_steps * self.planning_problem.goal.state_list[0].time_step.end)

        self.visualization_config = visualization_config
        self.initialize_run()

    @property
    def all_trajectories(self):
        return self.trajectory_planner.all_trajectories

    def initialize_run(self):
        """Initializes the run by planning the reference path and global trajectory"""
        self.current_state = StateVector(x=self.planning_problem.initial_state.position[0],
                                         y=self.planning_problem.initial_state.position[1],
                                         delta=0,
                                         v=self.planning_problem.initial_state.velocity,
                                         psi=self.planning_problem.initial_state.orientation)
        self.state_list.append(self.current_state)
        routes = self.route_planner.plan_routes()

        # plan reference path
        ref_path = ReferencePathPlanner(lanelet_network=self.scenario.lanelet_network,
                                        planning_problem=self.planning_problem,
                                        routes=routes).plan_shortest_reference_path(retrieve_shortest=True)
        # plan velocity profile with global trajectory
        vpp = VppBuilder().build_vpp(reference_path=ref_path, planning_problem=self.planning_problem,
                                     default_goal_velocity=self.planning_problem.initial_state.velocity)
        self.global_trajectory = self.velocity_planner.plan_velocity(
            reference_path=ref_path,
            planner_config=ConfigurationBuilder().get_predefined_configuration(),
            velocity_planning_problem=vpp)

        ref_path = np.hstack((ref_path.reference_path, ref_path.path_orientation.reshape(-1, 1)))

        # get road boundary
        poly = self.scenario.lanelet_network.lanelet_polygons[0].shapely_object
        for p in self.scenario.lanelet_network.lanelet_polygons:
            poly = poly.union(p.shapely_object)
        self.road_boundary = poly

        # update planner with external information
        self.trajectory_planner.update_externals(reference_path=ref_path,
                                                 current_road_boundary=poly)

    def run(self):
        """Runs the CommonRoad simulation"""
        next_state = self.current_state
        while self.timestep < self.max_time_steps_scenario:
            self.msg_logger.debug(f"current timestep {self.timestep}")
            self.current_state = next_state
            # get current vehicle state
            if len(self.cr_obstacle_list) > 0:
                ego_state = self.cr_obstacle_list[-1]
            else:
                shape = CRRectangle(self.trajectory_planner.vehicle_params.l, self.trajectory_planner.vehicle_params.w)
                ego_state = DynamicObstacle(self.planning_problem.planning_problem_id, ObstacleType.CAR, shape,
                                            self.planning_problem.initial_state)

            # get desired velocity at end of planning horizon
            position = np.array([self.current_state.x, self.current_state.y])

            horizon = self.trajectory_planner.horizon
            desired_velocity = self.global_trajectory.get_velocity_at_position_with_lookahead(position, horizon)
            self.msg_logger.debug(f"desired velocity: {desired_velocity}, current velocity: {self.current_state.v}")

            # check if goal reached
            self.goal_reached = self.planning_problem.goal.is_reached(ego_state.initial_state)
            if self.goal_reached:
                # simulation finished if goal is reached
                self.msg_logger.info("Goal reached")
                break

            # prepare planning step
            self.predictions = self.predictor.get_predictions(ego_state, self.timestep)

            self.trajectory_planner.update_externals(
                current_state=self.current_state,
                current_predictions=self.predictions,
                current_desired_velocity=desired_velocity
                )
            # plan next trajectories
            self.optimal_trajectory = self.trajectory_planner.plan()

            # add current trajectory to list
            self.cr_obstacle_list.append(self.convert_trajectory_to_commonroad_object(self.optimal_trajectory,
                                                                                      self.timestep))

            # visualize current timestep
            self.visualize_timestep(self.timestep)

            # prepare next iteration
            next_state = StateVector(x=self.optimal_trajectory['x'][1],
                                     y=self.optimal_trajectory['y'][1],
                                     v=self.optimal_trajectory['v'][1],
                                     psi=self.optimal_trajectory['psi'][1],
                                     delta=0
                                     )
            self.timestep += 1
            self.state_list.append(next_state)

        print("Simulation finished")

    def convert_trajectory_to_commonroad_object(self, optimal_trajectory, timestep):
        """
        Converts a trajectory to a CR dynamic obstacle with given dimensions
        :param state_list: trajectory state list of reactive planner
        :return: CR dynamic obstacle representing the ego vehicle
        """
        # shift trajectory positions to center
        obstacle_id: int = self.planning_problem.planning_problem_id
        new_state_list = list()
        wb_rear_axle = self.trajectory_planner.vehicle_params.b
        for idx in range(len(optimal_trajectory['x'])):
            position = np.array([optimal_trajectory['x'][idx] + wb_rear_axle * np.cos(optimal_trajectory['psi'][idx]),
                                 optimal_trajectory['y'][idx] + wb_rear_axle * np.sin(optimal_trajectory['psi'][idx])])
            state = CustomState(time_step=timestep + idx, position=position, orientation=optimal_trajectory['psi'][idx],
                                velocity=optimal_trajectory['v'][idx]
                                )

            new_state_list.append(state)

        trajectory = Trajectory(initial_time_step=new_state_list[0].time_step, state_list=new_state_list)
        # get shape of vehicle
        shape = CRRectangle(self.trajectory_planner.vehicle_params.l, self.trajectory_planner.vehicle_params.w)
        # get trajectory prediction
        prediction = TrajectoryPrediction(trajectory, shape)

        init_state = trajectory.state_list[0]
        initial_state = InitialState(position=init_state.position,
                                     orientation=init_state.orientation,
                                     velocity=init_state.velocity,
                                     time_step=init_state.time_step)

        return DynamicObstacle(obstacle_id, ObstacleType.CAR, shape, initial_state, prediction)

    def visualize_timestep(self, timestep):
        """Visualizes the current timestep"""
        if self.visualization_config.visualize:
            # get ego vehicle
            ego = self.cr_obstacle_list[timestep]
            ego_start_pos = ego.prediction.trajectory.state_list[0].position
            # create renderer object
            plot_limits = [-5 + ego_start_pos[0], 90 + ego_start_pos[0],
                           -7 + ego_start_pos[1], 18 + ego_start_pos[1]]
            rnd = MPRenderer(plot_limits=plot_limits, figsize=(10, 10))

            # set ego vehicle draw params
            ego_params = DynamicObstacleParams()
            ego_params.time_begin = timestep
            ego_params.draw_icon = True
            ego_params.show_label = False
            ego_params.vehicle_shape.occupancy.shape.facecolor = "#E37222"
            ego_params.vehicle_shape.occupancy.shape.edgecolor = "#9C4100"
            ego_params.vehicle_shape.occupancy.shape.zorder = 56
            ego_params.vehicle_shape.occupancy.shape.opacity = 1
            ego_params.trajectory.draw_trajectory = False

            # set obstacle draw params
            obs_params = MPDrawParams()
            obs_params.dynamic_obstacle.time_begin = timestep
            obs_params.dynamic_obstacle.draw_icon = True
            obs_params.dynamic_obstacle.show_label = False
            obs_params.dynamic_obstacle.trajectory.draw_trajectory = False
            obs_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#2266e3"
            obs_params.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = "#003359"
            obs_params.dynamic_obstacle.vehicle_shape.occupancy.shape.zorder = 56
            obs_params.dynamic_obstacle.zorder = 58
            obs_params.static_obstacle.show_label = True
            obs_params.static_obstacle.occupancy.shape.facecolor = "#a30000"
            obs_params.static_obstacle.occupancy.shape.edgecolor = "#756f61"

            self.scenario.draw(rnd, draw_params=obs_params)
            self.planning_problem.draw(rnd, draw_params=obs_params)
            ego.draw(rnd, draw_params=ego_params)
            rnd.render()

            if self.visualization_config.show_optimal_trajectory:
                # plot optimal trajectory
                rnd.ax.plot(self.optimal_trajectory['x'], self.optimal_trajectory['y'],
                            'kx-', markersize=1.5, zorder=21, linewidth=2.0)

            if self.visualization_config.show_all_trajectories:
                # plot all sampled trajectories that do not cross the road boundary (every 3rd trajectory)
                for i in range(0, len(self.all_trajectories), 5):
                    traj = self.all_trajectories[i]
                    points = np.array([traj['x'], traj['y']])
                    line = LineString(points.T)
                    if (self.road_boundary.contains_properly(line)):
                        rnd.ax.plot(traj['x'], traj['y'], color="#E37222", markersize=1.5,
                                    zorder=19, linewidth=1.0, alpha=0.5)

            if self.visualization_config.show_ref_path:
                # plot reference path
                rnd.ax.plot(self.global_trajectory.reference_path[:, 0], self.global_trajectory.reference_path[:, 1],
                            color='g', marker='.', markersize=1, zorder=21, linewidth=0.8,
                            label='reference path')

            # draw predictions
            draw_uncertain_predictions(self.predictions.predictions, rnd.ax)

            if self.visualization_config.make_gif:
                plot_dir = os.path.join(self.log_path, "plots")
                os.makedirs(plot_dir, exist_ok=True)
                plt.axis('off')
                plt.savefig(f"{plot_dir}/{self.scenario.scenario_id}_{timestep}.png", format='png', dpi=300,
                            bbox_inches='tight', pad_inches=0)
            # show plot
            if self.visualization_config.render_plots:
                matplotlib.use("TkAgg")
                plt.pause(0.0001)

    def plot_final_trajectory(self):
        """
        Function plots occupancies for a given CommonRoad trajectory (of the ego vehicle)
        """
        # create renderer object
        plot_limits = [20, 125, -5, 10]
        rnd = MPRenderer(plot_limits=plot_limits, figsize=(20, 4))

        # set renderer draw params
        rnd.draw_params.time_begin = 0
        rnd.draw_params.planning_problem.initial_state.state.draw_arrow = False
        rnd.draw_params.planning_problem.initial_state.state.radius = 0.5

        # visualize scenario
        self.scenario.lanelet_network.draw(rnd)

        # obtacles
        # set occupancy shape params
        occ_params = ShapeParams()
        occ_params.facecolor = "#2266e3"
        occ_params.edgecolor = "#003359"
        occ_params.opacity = 1.0
        occ_params.zorder = 51

        obs_params = DynamicObstacleParams()
        obs_params.time_begin = 0
        obs_params.draw_icon = True
        obs_params.show_label = False
        obs_params.vehicle_shape.occupancy.shape.facecolor = "#2266e3"
        obs_params.vehicle_shape.occupancy.shape.edgecolor = "#003359"
        obs_params.zorder = 52
        obs_params.trajectory.draw_trajectory = False

        trajs = []
        for ob in self.scenario.dynamic_obstacles:
            traj = []
            # draw bounding boxes along trajectory
            for idx_p, pred in enumerate(ob.prediction.occupancy_set):
                occ_pos = pred.shape
                traj.append(occ_pos.center)
                if idx_p % 2 == 0:
                    occ_params.opacity = 0.3
                    occ_params.zorder = 50
                    occ_pos.draw(rnd, draw_params=occ_params)
            traj = np.array(traj)
            trajs.append(traj)
            # draw initial obstacle
            ob.draw(rnd, draw_params=obs_params)

        # plot final ego trajectory
        occ_params = ShapeParams()
        occ_params.facecolor = '#E37222'
        occ_params.edgecolor = '#9C4100'
        occ_params.opacity = 1.0
        occ_params.zorder = 51

        for idx, obs in enumerate(self.cr_obstacle_list):
            # plot bounding boxes along trajectory
            state = obs.initial_state
            occ_pos = ShapeRectangle(length=self.trajectory_planner.vehicle_params.l,
                                     width=self.trajectory_planner.vehicle_params.w,
                                     center=state.position,
                                     orientation=state.orientation)
            if idx >= 1:
                occ_params.opacity = 0.3
                occ_params.zorder = 50
                occ_pos.draw(rnd, draw_params=occ_params)

        ego_params = DynamicObstacleParams()
        ego_params.time_begin = 0
        ego_params.draw_icon = True
        ego_params.show_label = False
        ego_params.vehicle_shape.occupancy.shape.facecolor = "#E37222"
        ego_params.vehicle_shape.occupancy.shape.edgecolor = "#9C4100"
        ego_params.zorder = 52
        ego_params.trajectory.draw_trajectory = False
        # plot initial ego vehicle
        self.cr_obstacle_list[0].draw(rnd, draw_params=ego_params)
        # plot initial optimal trajectory
        rbfn_traj = [state.position for state in self.cr_obstacle_list[0].prediction.trajectory.state_list]
        rbfn_traj = np.array(rbfn_traj)

        # render scenario and occupancies
        rnd.render()

        # visualize ego trajectory
        pos = np.asarray([obs.initial_state.position for obs in self.cr_obstacle_list])
        rnd.ax.plot(pos[::5, 0,], pos[::5, 1,], color='k', marker='|', markersize=7.0,
                    markeredgewidth=0.4, zorder=21, linewidth=0.8)
        rnd.ax.plot(rbfn_traj[:, 0], rbfn_traj[:, 1], color="#783505", linestyle='dashed', zorder=21, linewidth=1.2)

        # visualize other obstacles' trajectories
        for traj in trajs:
            rnd.ax.plot(traj[::5, 0], traj[::5, 1], color='k', marker='|', markersize=7, zorder=21, linewidth=0.8)

        # show plot
        if self.visualization_config.render_plots:
            matplotlib.use("TkAgg")
            plt.show()

    def create_gif(self):
        if self.visualization_config.make_gif:
            images = []
            filenames = []

            # directory, where single images are outputted (see visualize_planner_at_timestep())
            path_images = os.path.join(self.log_path, "plots")

            for step in range(self.timestep):
                im_path = os.path.join(path_images, str(self.scenario.scenario_id) + f"_{step}.png")
                filenames.append(im_path)

            img_width = 0
            img_height = 0

            for filename in filenames:
                img = Image.open(filename)
                width, height = img.size

                if not img_width:
                    img_width = width
                    img_height = height

                # Calculate the area to crop
                left = 0
                top = 0
                right = img_width if width != img_width else width
                bottom = img_height if height != img_height else height

                # Crop the image
                img_cropped = img.crop((left, top, right, bottom))

                # Convert the PIL image to an array for imageio
                img_array = np.array(img_cropped)
                images.append(img_array)

            iio.imwrite(os.path.join(self.log_path, str(self.scenario.scenario_id) + ".gif"),
                        images, duration=self.trajectory_planner.dT)
