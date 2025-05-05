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
import torch

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.draw_params import (
    MPDrawParams,
    DynamicObstacleParams,
    ShapeParams,
)
from commonroad.geometry.shape import Rectangle as CRRectangle
from commonroad_route_planner.route_planner import RoutePlanner
from commonroad_route_planner.reference_path_planner import ReferencePathPlanner
from commonroad_velocity_planner.velocity_planner_interface import IVelocityPlanner
from commonroad_velocity_planner.configuration.configuration_builder import (
    ConfigurationBuilder,
)
from commonroad_velocity_planner.velocity_planning_problem import VppBuilder
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.geometry.shape import Rectangle as ShapeRectangle
from commonroad.scenario.state import CustomState, InitialState
from commonroad.scenario.trajectory import Trajectory

from ml_planner.planner.ml_planner import MLPlanner
from ml_planner.general_utils.data_types import DTYPE, StateTensor
from ml_planner.general_utils.logging import logger_initialization

from .utils.predictions import Predictor
from .utils.visualization_utils import (
    draw_uncertain_predictions,
    EgoPatchHandler,
    ObsPatchHandler,
    TrajPatchHandler,
)
from ml_planner.planner.planner_utils import (
    create_cov_matrix,
    convert_globals_to_locals,
)


class CommonroadInterface(ABC):
    """Interface between CommonRoad and MP-RBFN planner"""

    def __init__(
        self,
        planner_config,
        prediction_config,
        visualization_config,
        uncertainty_config,
        interface_logging_level,
        log_path,
        max_steps,
        scenario_path,
    ):
        super().__init__()
        self.log_path = log_path
        self.msg_logger = logger_initialization(
            log_path=log_path,
            logger="Interface_Logger",
            loglevel=interface_logging_level.upper(),
        )
        # load scenario and planning problem
        self.scenario, self.planning_problem_set = CommonRoadFileReader(scenario_path).open()
        self.planning_problem = list(self.planning_problem_set.planning_problem_dict.values())[0]

        # initialize route planner
        self.route_planner = RoutePlanner(
            lanelet_network=self.scenario.lanelet_network,
            planning_problem=self.planning_problem,
            scenario=self.scenario,
            extended_search=False,
        )

        # initialize velocity planner
        self.velocity_planner = IVelocityPlanner()

        # initialize MP-RBFN sampling planner
        trajectory_planner = cast(
            MLPlanner,
            instantiate(planner_config, log_path=log_path),
        )
        assert trajectory_planner.dt == self.scenario.dt, "trajectory_planner.dt and scenario.dt must be equal"

        self.device = trajectory_planner.device
        self.trajectory_planner = trajectory_planner.to(self.device)
        # initialize prediction module
        self.predictor = Predictor(prediction_config, self.scenario, self.trajectory_planner.horizon)

        # initialize variables
        self.reference_trajectory = None
        # self.road_boundary = None
        self.goal_reached = False
        # self.ini_state = None
        self.optimal_trajectory = None
        self.planner_state_list = []
        self.cr_obstacle_list = []
        self.obstacle_predictions_global = None
        self.timestep = 0
        self.max_time_steps_scenario = int(max_steps * self.planning_problem.goal.state_list[0].time_step.end)

        self.uncertainty = uncertainty_config
        self.visualization_config = visualization_config
        self.initialize_run()

    @property
    def all_trajectories(self):
        return self.trajectory_planner.all_trajs_global

    def initialize_run(self):
        """Initializes the run by planning the reference path and global trajectory"""
        # Origin of ML Planner: rear axis!
        wb_rear_axle = self.trajectory_planner.vehicle_params.b
        psi = self.planning_problem.initial_state.orientation
        delta = (
            self.planning_problem.initial_state.delta if hasattr(self.planning_problem.initial_state, "delta") else 0.0
        )
        initial_state = StateTensor(
            {
                "x": self.planning_problem.initial_state.position[0] - wb_rear_axle * np.cos(psi),
                "y": self.planning_problem.initial_state.position[1] - wb_rear_axle * np.sin(psi),
                "psi": psi,
                "v": self.planning_problem.initial_state.velocity,
                "delta": delta,
            },
            device=self.device,
        )
        # convert localization uncertainty from config from local to global
        local_covs = create_cov_matrix(initial_state, self.uncertainty.localization)
        rot_matrix = torch.eye(local_covs.shape[-1], dtype=DTYPE, device=initial_state.device)
        rot_matrix[:2, :2] = torch.tensor(
            [
                [torch.cos(initial_state.psi), torch.sin(initial_state.psi)],
                [-torch.sin(initial_state.psi), torch.cos(initial_state.psi)],
            ],
            dtype=DTYPE,
            device=initial_state.device,
        )
        global_covs = rot_matrix.T @ local_covs @ rot_matrix

        initial_state.add_cov(global_covs)

        self.planner_state_list.append(initial_state)
        # origin of CR Vehicle: COG
        self.cr_obstacle_list.append(
            DynamicObstacle(
                self.planning_problem.planning_problem_id,
                ObstacleType.CAR,
                CRRectangle(
                    self.trajectory_planner.vehicle_params.l,
                    self.trajectory_planner.vehicle_params.w,
                ),
                self.planning_problem.initial_state,
            )
        )

        routes = self.route_planner.plan_routes()

        # plan reference path
        ref_path = ReferencePathPlanner(
            lanelet_network=self.scenario.lanelet_network,
            planning_problem=self.planning_problem,
            routes=routes,
        ).plan_shortest_reference_path(retrieve_shortest=True)
        # plan velocity profile with global trajectory
        vpp = VppBuilder().build_vpp(
            reference_path=ref_path,
            planning_problem=self.planning_problem,
            default_goal_velocity=self.planning_problem.initial_state.velocity,
        )
        self.reference_trajectory = self.velocity_planner.plan_velocity(
            reference_path=ref_path,
            planner_config=ConfigurationBuilder().get_predefined_configuration(),
            velocity_planning_problem=vpp,
        )

        path = torch.tensor(np.hstack((ref_path.reference_path, ref_path.path_orientation.reshape(-1, 1))), dtype=DTYPE)

        self.ref_path_global = StateTensor(
            states=path,
            device=self.device,
        )

        # get road boundary: needed if no road boundary of visible area is provided
        poly = self.scenario.lanelet_network.lanelet_polygons[0].shapely_object
        for p in self.scenario.lanelet_network.lanelet_polygons:
            poly = poly.union(p.shapely_object)
        self.global_road_boundary = poly

    def run(self):
        """Runs the CommonRoad simulation"""
        cr_state_global = self.cr_obstacle_list.pop()
        self.max_time_steps_scenario = 2
        while self.timestep < self.max_time_steps_scenario:
            self.msg_logger.debug(f"current timestep {self.timestep}")
            # check if goal reached
            self.goal_reached = self.planning_problem.goal.is_reached(cr_state_global.initial_state)
            if self.goal_reached:
                # simulation finished if goal is reached
                self.msg_logger.info("Goal reached")
                break

            self.plan_step(cr_state_global)

            # add current trajectory to list
            cr_state_global = self.convert_trajectory_to_commonroad_object(self.optimal_trajectory, self.timestep)
            self.cr_obstacle_list.append(cr_state_global)

            # visualize current timestep
            self.visualize_timestep(self.timestep)

            # prepare next iteration
            next_state = StateTensor(
                states=self.optimal_trajectory.states[1],
                covs=self.optimal_trajectory.covs[1],
                device=self.device,
            )

            self.timestep += 1
            self.planner_state_list.append(next_state)

        self.msg_logger.info("Simulation finished")

    def plan_step(self, cr_state_global):
        planner_state_global = self.planner_state_list[-1]
        # get desired velocity at end of planning horizon
        desired_velocity = self.reference_trajectory.get_velocity_at_position_with_lookahead(
            position=planner_state_global.coords,
            lookahead_s=self.trajectory_planner.horizon,
        )
        self.msg_logger.debug(f"desired velocity: {desired_velocity}, current velocity: {planner_state_global.v}")

        # prepare planning step
        # update local reference path
        ref_path_local = convert_globals_to_locals(self.ref_path_global, planner_state_global)

        # get global predictions and road boundary
        self.obstacle_predictions_global, visible_area_global = self.predictor.get_predictions(
            cr_state_global, self.timestep
        )
        current_visible_area_global = visible_area_global if visible_area_global else self.global_road_boundary
        # convert global predictions to local
        obstacle_predictions_local = self.obstacle_predictions_global.create_StateTensors(self.device)
        for pred in obstacle_predictions_local:
            pred["states"] = convert_globals_to_locals(pred["states"], planner_state_global)

        # convert global road boundary to local
        x, y = current_visible_area_global.segmentize(0.2).exterior.xy

        current_visible_area_global = StateTensor(states={"x": x, "y": y}, device=self.device)
        current_visible_area_local = convert_globals_to_locals(current_visible_area_global, planner_state_global)
        current_visible_area_local.add_cov(create_cov_matrix(current_visible_area_local, self.uncertainty.boundary))

        lane_predictions_local = [
            {"id": 0, "states": current_visible_area_local, "types": {"wall": 1}, "boundary": True}
        ]

        # plan next trajectories
        self.optimal_trajectory = self.trajectory_planner.plan(
            current_state_global=planner_state_global,
            desired_velocity=desired_velocity,
            reference_path_local=ref_path_local,
            obstacle_predictions_local=obstacle_predictions_local,
            lane_predictions_local=lane_predictions_local,
        )

    def convert_trajectory_to_commonroad_object(self, trajectory, timestep):
        """
        Converts a trajectory to a CR dynamic obstacle with given dimensions
        :param state_list: trajectory state list of reactive planner
        :return: CR dynamic obstacle representing the ego vehicle
        """

        # create trajectory and shift positions to center of vehicle
        obstacle_id: int = self.planning_problem.planning_problem_id
        wb_rear_axle = self.trajectory_planner.vehicle_params.b

        shift = torch.vstack([wb_rear_axle * torch.cos(trajectory.psi), wb_rear_axle * torch.sin(trajectory.psi)]).T
        pos = trajectory.coords + shift
        pos = pos.detach().cpu().numpy()
        orientation = trajectory.psi.detach().cpu()
        velocity = trajectory.v.detach().cpu()
        delta = trajectory.delta.detach().cpu() if hasattr(trajectory, "delta") else np.zeros_like(velocity)
        state_list = [
            CustomState(
                time_step=timestep + idx,
                position=pos[idx],
                orientation=float(orientation[idx]),
                velocity=float(velocity[idx]),
                delta=float(delta[idx]),
            )
            for idx in range(trajectory.num_states)
        ]

        CRtrajectory = Trajectory(initial_time_step=timestep, state_list=state_list)
        # get shape of vehicle
        shape = CRRectangle(
            self.trajectory_planner.vehicle_params.l,
            self.trajectory_planner.vehicle_params.w,
        )
        # get trajectory prediction
        prediction = TrajectoryPrediction(CRtrajectory, shape)

        init_state = CRtrajectory.state_list[0]
        initial_state = InitialState(
            position=init_state.position,
            orientation=init_state.orientation,
            velocity=init_state.velocity,
            time_step=init_state.time_step,
        )

        return DynamicObstacle(obstacle_id, ObstacleType.CAR, shape, initial_state, prediction)

    def visualize_timestep(self, timestep):
        """Visualizes the current timestep"""
        if self.visualization_config.make_plot or self.visualization_config.make_gif:
            # get ego vehicle
            ego = self.cr_obstacle_list[timestep]
            ego_start_pos = ego.initial_state.position
            # create renderer object
            plot_limits = [
                -30 + ego_start_pos[0],
                90 + ego_start_pos[0],
                -30 + ego_start_pos[1],
                90 + ego_start_pos[1],
            ]
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
            if self.visualization_config.show_optimal_trajectory:
                # plot optimal trajectory
                ego_params.trajectory.draw_trajectory = True
                ego_params.trajectory.draw_continuous = True
                ego_params.trajectory.line_width = 2.0
            else:
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
            self.scenario.lanelet_network.draw(rnd)
            self.scenario.draw(rnd, draw_params=obs_params)
            self.planning_problem.draw(rnd, draw_params=obs_params)
            ego.draw(rnd, draw_params=ego_params)
            rnd.render()

            if self.visualization_config.show_all_trajectories:
                # plot all sampled trajectories that do not cross the road boundary (every 3rd trajectory)
                for i in range(0, self.all_trajectories.num_batches, 3):
                    rnd.ax.plot(
                        self.all_trajectories.coords[i, :, 0],
                        self.all_trajectories.coords[i, :, 1],
                        color="#E37222",
                        markersize=1.5,
                        zorder=19,
                        linewidth=1.0,
                        alpha=0.5,
                    )

            if self.visualization_config.show_ref_path:
                # plot reference path
                rnd.ax.plot(
                    self.reference_trajectory.reference_path[:, 0],
                    self.reference_trajectory.reference_path[:, 1],
                    color="g",
                    marker=".",
                    markersize=1,
                    zorder=21,
                    linewidth=0.8,
                    label="reference path",
                )

            # draw predictions
            draw_uncertain_predictions(self.obstacle_predictions_global.predictions, rnd.ax)
            plot_dir = os.path.join(self.log_path, "plots")
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(
                f"{plot_dir}/{self.scenario.scenario_id}_{timestep}.svg",
                format="svg",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,
                transparent=False,
            )
            if self.visualization_config.make_gif:
                plt.axis("off")
                plt.savefig(
                    f"{plot_dir}/{self.scenario.scenario_id}_{timestep}.png",
                    format="png",
                    dpi=300,
                    bbox_inches="tight",
                    pad_inches=0,
                )
            # show plot
            if self.visualization_config.render_plots:
                matplotlib.use("TkAgg")
                plt.pause(0.0001)

    def plot_final_trajectory(self):
        """
        Function plots occupancies for a given CommonRoad trajectory (of the ego vehicle)
        """
        # create renderer object
        # plot_limits = [20, 125, -5, 10]
        ego = self.cr_obstacle_list[0]
        positions = np.array([i.initial_state.position for i in self.cr_obstacle_list])
        min_pos = np.min(positions, axis=0)
        max_pos = np.max(positions, axis=0)
        plot_limits = [
            -10 + min_pos[0],
            10 + max_pos[0],
            -10 + min_pos[1],
            10 + max_pos[1],
        ]
        rnd = MPRenderer(plot_limits=plot_limits, figsize=(15, 15))  # (20, 4))

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
                if idx_p > self.max_time_steps_scenario:
                    break
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
        occ_params.facecolor = "#E37222"
        occ_params.edgecolor = "#9C4100"
        occ_params.opacity = 1.0
        occ_params.zorder = 51

        for idx, obs in enumerate(self.cr_obstacle_list):
            # plot bounding boxes along trajectory
            state = obs.initial_state
            occ_pos = ShapeRectangle(
                length=self.trajectory_planner.vehicle_params.l,
                width=self.trajectory_planner.vehicle_params.w,
                center=state.position,
                orientation=state.orientation,
            )
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
        # plot initial optimal trajectory, first index does not have a prediction
        rbfn_traj = [state.position for state in self.cr_obstacle_list[0].prediction.trajectory.state_list]
        rbfn_traj = np.array(rbfn_traj)

        # render scenario and occupancies
        rnd.render()

        # visualize ego trajectory
        pos = np.asarray([obs.initial_state.position for obs in self.cr_obstacle_list])
        rnd.ax.plot(
            pos[
                ::5,
                0,
            ],
            pos[
                ::5,
                1,
            ],
            color="k",
            marker="|",
            markersize=7.0,
            markeredgewidth=0.4,
            zorder=21,
            linewidth=0.8,
            label="Final trajectory",
        )

        # visualize other obstacles' trajectories
        for traj in trajs:
            rnd.ax.plot(
                traj[::5, 0],
                traj[::5, 1],
                color="k",
                marker="|",
                markersize=7,
                zorder=21,
                linewidth=0.8,
            )

        # Add legend
        obs_patch = matplotlib.patches.Patch(facecolor="#2266e3", edgecolor="#003359", label="Obstacle vehicle")
        ego_patch = matplotlib.patches.Patch(facecolor="#E37222", edgecolor="#9C4100", label="Ego vehicle")
        traj_patch = matplotlib.lines.Line2D(
            [], [], color="k", marker="|", markersize=7.0, linewidth=0.8, label="Final trajectory"
        )

        rnd.ax.legend(
            handles=[ego_patch, obs_patch, traj_patch],  # , rbfn_traj[0]],
            handler_map={ego_patch: EgoPatchHandler(), obs_patch: ObsPatchHandler(), traj_patch: TrajPatchHandler()},
            loc="upper left",
        )

        plot_dir = os.path.join(self.log_path, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(
            f"{plot_dir}/final_trajectory.svg",
            format="svg",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
            transparent=False,
        )

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

            iio.imwrite(
                os.path.join(self.log_path, str(self.scenario.scenario_id) + ".gif"),
                images,
                duration=self.trajectory_planner.dt,
            )
