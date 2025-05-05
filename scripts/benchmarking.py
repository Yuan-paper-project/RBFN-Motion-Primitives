__author__ = "Marc Kaufed"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

from pathlib import Path
import inspect
from typing import Callable
import pandas as pd
import hydra
import timeit
import numpy as np
from copy import deepcopy
import torch
import matplotlib
import matplotlib.pyplot as plt
from ml_planner.general_utils.data_types import DTYPE
from ml_planner.sampling.data_loader import DataSetAllLabels, DataSetStatesOnly, make_splits
from ml_planner.sampling.sampling_network_utils import load_model
from ml_planner.general_utils.logging import logger_initialization
from commonroad_route_planner.reference_path_planner import ReferencePathPlanner
from commonroad_route_planner.route_planner import RoutePlanner
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.draw_params import MPDrawParams, DynamicObstacleParams

from frenetix_motion_planner.polynomial_trajectory import QuinticTrajectory, QuarticTrajectory
from frenetix_motion_planner.reactive_planner import ReactivePlannerPython
from frenetix_motion_planner.reactive_planner_cpp import ReactivePlannerCpp
from cr_scenario_handler.utils.configuration_builder import ConfigurationBuilder
from frenetix_motion_planner.trajectories import TrajectorySample
from frenetix_motion_planner.state import ReactivePlannerState


from ml_planner.simulation_interfaces.commonroad.utils.visualization_utils import (
    EgoPatchHandler,
    ObsPatchHandler,
    TrajPatchHandler,
)

# from run_cr_simulation import create_config
from ml_planner.simulation_interfaces.commonroad.commonroad_interface import CommonroadInterface


tumblue = "#0065BD"
darkblue = "#005293"
orange = "#E37222"
green = "#A2AD00"

def test_accuracy(models):
    cwd = Path.cwd()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")

    dataset_dir = "dataset_BMW320i_kinematic_single_track_steering_jerk_full_v1"

    dataset: Callable = DataSetStatesOnly

    data_dir = cwd.parent / "dataset" / dataset_dir  # / dataset_name

    dataset = dataset(data_dir)

    dataloader_kwargs: dict = {
        "train_split": 0.7,
        "batch_size": 12000,
        "use_bounds_in_training": True,
        "shuffle": True,
        "num_workers": 10,
        "persistent_workers": True,
    }
    train_dataloader, test_dataloader = make_splits(dataset, dataloader_kwargs, None)

    data_idx = test_dataloader.dataset.indices
    data_all = dataset.data[data_idx]
    # Test Cases
    v_in_idx = test_dataloader.dataset.dataset.labels.index("v")
    x_in_idx = test_dataloader.dataset.dataset.labels.index("x")
    y_in_idx = test_dataloader.dataset.dataset.labels.index("y")
    psi_in_idx = test_dataloader.dataset.dataset.labels.index("psi")
    delta_in_idx = test_dataloader.dataset.dataset.labels.index("delta")

    psi_measure = True
    if psi_measure:
        values = [0, 0.8, 1.6]
    else:
        values = [-1]
    for j in values:
        if psi_measure:
            mask = data_all[:, -1, psi_in_idx].round(decimals=2) == round(j, 2)
            data = data_all[mask]
        else:
            data = data_all[:j]
        if data.shape[0] == 0:
            raise ValueError("No data for psi_f")
        gt_trajs = data.cpu().numpy()

        y = data[:, -1, y_in_idx].round().reshape(-1, 1).type(DTYPE)
        x = data[:, -1, x_in_idx].round().reshape(-1, 1).type(DTYPE)
        psi = data[:, -1, psi_in_idx].reshape(-1, 1).type(DTYPE)
        v = data[:, 0, v_in_idx].reshape(-1, 1).type(DTYPE)
        delta = data[:, 0, delta_in_idx].reshape(-1, 1).type(DTYPE)

        x0_loc = torch.tensor([0.0, 0.0, 0.0], dtype=DTYPE).repeat(y.shape[0], 1)
        input_x = torch.cat([x0_loc, v, delta, x, y, psi], dim=1).to(device)

        results = {}
        for nnname, nntype in models.items():
            with torch.no_grad():
                path = cwd / "ml_planner" / "sampling" / "models" / nnname
                network, _ = load_model(path, nntype, device)
                x_out_idx = network.output_labels.index("x")
                y_out_idx = network.output_labels.index("y")
                v_out_idx = network.output_labels.index("v")
                psi_out_idx = network.output_labels.index("psi")
                delta_out_idx = network.output_labels.index("delta")
                network = network.to(device)
                network.eval()
                pred = network(input_x).cpu().detach().numpy()
                pos = np.sqrt(
                    (pred[:, :, x_out_idx] - gt_trajs[:, :, x_in_idx]) ** 2
                    + (pred[:, :, y_out_idx] - gt_trajs[:, :, y_in_idx]) ** 2
                )
                vel = np.sqrt((pred[:, :, v_out_idx] - gt_trajs[:, :, v_in_idx]) ** 2)
                ori = np.sqrt((pred[:, :, psi_out_idx] - gt_trajs[:, :, psi_in_idx]) ** 2)
                delta = np.sqrt((pred[:, :, delta_out_idx] - gt_trajs[:, :, delta_in_idx]) ** 2)
                num_rep = 1000
                time = []
                for i in range(num_rep):
                    time.append(timeit.timeit(lambda: network(input_x), number=1))
                results[nnname] = {
                    "pos": pos,
                    "vel": vel,
                    "ori": ori,
                    "delta": delta,
                    "time": np.array(time),
                }

        print(f"{j}:\n")
        for nnname, result in results.items():
            print(f"{nnname}:\n")
            print(f"Position Error: Mean {result['pos'].mean()}, Max: {result['pos'].max()}")
            print(f"Velocity Error: Mean {result['vel'].mean()}, Max: {result['vel'].max()}")
            print(f"Orientation Error: Mean {result['ori'].mean()}, Max: {result['ori'].max()}")
            print(f"Delta Error: Mean {result['delta'].mean()}, Max: {result['delta'].max()}")
            print(f"Time: Mean {result['time'].mean()*1e3}, Max: {result['time'].max()*1e3}\n")


def test_frenetix():
    cwd = Path.cwd()
    dataset_dir = "dataset_BMW320i_kinematic_single_track_steering_jerk_full_v1"

    dataset: Callable = DataSetStatesOnly

    data_dir = cwd.parent / "dataset" / dataset_dir  # / dataset_name

    dataset = dataset(data_dir)

    dataloader_kwargs: dict = {
        "train_split": 0.7,
        "batch_size": 12000,
        "use_bounds_in_training": True,
        "shuffle": True,
        "num_workers": 10,
        "persistent_workers": True,
    }
    train_dataloader, test_dataloader = make_splits(dataset, dataloader_kwargs, None)

    data_idx = test_dataloader.dataset.indices
    data_all = dataset.data[data_idx]

    DATA_PATH = cwd / "example_scenarios"
    scenario_path = DATA_PATH / "ZAM_Tjunction-1_42_T-1.xml"
    scenario, planning_problem_set = CommonRoadFileReader(scenario_path).open()
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

    # Test Cases
    x_idx = dataset.labels.index("x")
    y_idx = dataset.labels.index("y")
    psi_idx = dataset.labels.index("psi")
    v_idx = dataset.labels.index("v")
    # frenetix
    mod_path = Path(inspect.getfile(ReactivePlannerPython)).parent.parent
    config_sim = ConfigurationBuilder.build_sim_configuration("abc", "abc", str(mod_path))
    config_planner = ConfigurationBuilder.build_frenetplanner_configuration("abc")
    config_planner.debug.use_cpp = True
    config_planner.debug.multiproc = False

    msg_logger = logger_initialization(log_path=str(DATA_PATH), logger="Frenetix", loglevel="DEBUG")
    frenetix = ReactivePlannerPython(
        config_planner, config_sim, scenario, planning_problem, str(DATA_PATH), mod_path, msg_logger
    )
    x_ref = np.linspace(-10, 160, 340)
    y_ref = np.zeros_like(x_ref)
    ref_path = np.vstack((x_ref, y_ref)).T
    frenetix.set_reference_and_coordinate_system(ref_path)

    data_all = data_all[abs(data_all[:, -1, y_idx]) <= 15]
    for psi_f in [0, 0.8, 1.6]:

        data = data_all.numpy()
        if data.shape[0] == 0:
            raise ValueError("No data for psi_f")
        trajs = []
        pos_errors = []
        vel_errors = []
        or_errors = []
        trajs_valid = 0
        for i in range(data.shape[0]):
            try:
                s0, d0 = frenetix.coordinate_system.convert_to_curvilinear_coords(0, 0)
                sd, dd = frenetix.coordinate_system.convert_to_curvilinear_coords(
                    data[i, -1, x_idx], data[i, -1, y_idx]
                )
            except:
                continue
            v0 = data[i, 0, v_idx]
            vd = data[i, -1, v_idx]
            x0_long = np.array([s0, v0, 0])
            x_d_long = np.array([sd, vd, 0])
            frenetix.x_0 = ReactivePlannerState(0, np.array([0, 0]), 0, v0, 0, 0)
            traj_long = QuinticTrajectory(tau_0=0, delta_tau=3, x_0=x0_long, x_d=x_d_long)

            x0_lat = np.array([d0, 0, 0])
            x_d_lat = np.array([dd, 0, 0])
            traj_lat = QuinticTrajectory(tau_0=0, delta_tau=3, x_0=x0_lat, x_d=x_d_lat)
            trajectory = TrajectorySample(
                frenetix.horizon, frenetix.dT, traj_long, traj_lat, 0, frenetix.cost_function.cost_weights
            )
            traj = frenetix.check_feasibility([trajectory])
            if traj[0].valid and traj[0].feasible:
                cart = traj[0].cartesian
                dx = data[i, :, x_idx] - cart.x
                dy = data[i, :, y_idx] - cart.y
                pos_errors.append(np.sqrt(dx**2 + dy**2).mean())
                vel_errors.append(np.sqrt((data[i, :, v_idx] - cart.v) ** 2).mean())
                or_errors.append(np.sqrt((data[i, :, psi_idx] - cart.theta) ** 2).mean())
                trajs_valid += 1
                trajs.append(cart)

        print(f"psi_f: {psi_f}")
        print(trajs_valid)
        print(f"Valid Trajectories: {trajs_valid/data.shape[0]}")
        pos_errors = np.array(pos_errors)
        vel_errors = np.array(vel_errors)
        or_errors = np.array(or_errors)
        print(f"Position Error: Mean {pos_errors.mean()}, Max: {pos_errors.max()}")
        print(f"Velocity Error: Mean {vel_errors.mean()}, Max: {vel_errors.max()}")
        print(f"Orientation Error: Mean {or_errors.mean()}, Max: {or_errors.max()}")


def vis_comparison(models):
    CWD = Path.cwd()
    DATA_PATH = CWD / "example_scenarios"
    SCENARIO = DATA_PATH / "ZAM_Tjunction-1_42_T-1_custom.xml"
    LOG_PATH = CWD / "logs"

    MODEL_PATH = CWD / "ml_planner" / "sampling" / "models"

    LOGGING_LEVEL_INTERFACE = "debug"
    LOGGING_LEVEL_PLANNER = "debug"

    def create_config():
        """
        Creates a configuration for the simulation.

        Returns:
            The configuration object for the simulation.
        """
        # config overrides
        overrides = [
            # f"mod_path={CWD}",
            # general overrides
            f"log_path= {LOG_PATH}",
            # simulation overrides
            f"interface_logging_level={LOGGING_LEVEL_INTERFACE}",
            f"scenario_path={SCENARIO}",
            # planner overrides
            f"planner_config.logging_level={LOGGING_LEVEL_PLANNER}",
            f"planner_config.sampling_model_path={MODEL_PATH}",
        ]

        # Compose the configuration
        config_dir = str(Path.cwd() / "ml_planner" / "simulation_interfaces" / "commonroad" / "configurations")
        with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
            config = hydra.compose(config_name="simulation", overrides=overrides)
        return config

    config = create_config()
    # create simulation interface
    interface = CommonroadInterface(**config)
    cr_state_global = interface.cr_obstacle_list[0]
    interface.plan_step(cr_state_global)

    all_trajs = interface.all_trajectories
    traj_states = deepcopy(all_trajs.states)
    traj_states = traj_states[traj_states[..., -1, all_trajs.psi_idx] > -0.5]
    traj_states = traj_states[traj_states[:, -1, all_trajs.y_idx] <= 12]
    traj_states = traj_states[traj_states[:, -1, all_trajs.x_idx] <= 28]
    traj_states = traj_states[traj_states[:, -1, all_trajs.x_idx] >= 11]
    traj_states = traj_states[
        ~((traj_states[..., -1, all_trajs.psi_idx] < 0.8) & (traj_states[..., -1, all_trajs.y_idx] > 5))
    ]

    traj_states = traj_states[
        ~((traj_states[..., -1, all_trajs.psi_idx] > 0.8) & (traj_states[..., -1, all_trajs.y_idx] < 5))
    ]

    scenario = interface.scenario
    planning_problem = interface.planning_problem

    # frenetix
    mod_path = Path(inspect.getfile(ReactivePlannerPython)).parent.parent
    config_sim = ConfigurationBuilder.build_sim_configuration("abc", "abc", str(mod_path))
    config_planner = ConfigurationBuilder.build_frenetplanner_configuration("abc")
    config_planner.debug.multiproc = False
    config_sim.simulation.ego_agent_id = 42

    msg_logger = logger_initialization(log_path=str(DATA_PATH), logger="Frenetix", loglevel="DEBUG")
    frenetix = ReactivePlannerCpp(
        config_planner, config_sim, scenario, planning_problem, str(DATA_PATH), mod_path, msg_logger
    )
    frenetix.set_reference_and_coordinate_system(interface.reference_trajectory.reference_path)

    x_0 = ReactivePlannerState.create_from_initial_state(
        deepcopy(planning_problem.initial_state), config_sim.vehicle.wheelbase, config_sim.vehicle.wb_rear_axle
    )

    frenetix.update_externals(x_0=x_0, desired_velocity=x_0.velocity)
    x0_long = frenetix.x_cl[0]
    x0_lat = frenetix.x_cl[1]
    frenet_trajs = []
    for final_state in traj_states[:, -1, :].tolist():
        try:
            x = final_state[all_trajs.x_idx]
            y = final_state[all_trajs.y_idx]
            v = final_state[all_trajs.v_idx]
            sf, df = frenetix.coordinate_system.convert_to_curvilinear_coords(x, y)
            xf_long = np.array([sf, v, 0])
            xf_lat = np.array([df, 0, 0])
            traj_long = QuarticTrajectory(tau_0=0, delta_tau=3, x_0=np.array(x0_long), x_d=np.array(xf_long[1:]))
            # traj_long = QuinticTrajectory(tau_0=0, delta_tau=3, x_0=np.array(x0_long), x_d=np.array(xf_long))

            traj_lat = QuinticTrajectory(tau_0=0, delta_tau=3, x_0=np.array(x0_lat), x_d=np.array(xf_lat))
            trajectory = TrajectorySample(
                frenetix.horizon, frenetix.dT, traj_long, traj_lat, 0, frenetix.cost_function.cost_weights
            )
            frenet_trajs.append(frenetix.check_feasibility([trajectory])[0])
            pass
        except:
            pass

    def draw(traj_states):
        # visualization
        obs_params = MPDrawParams()
        obs_params.axis_visible = False
        obs_params.dynamic_obstacle.draw_icon = True
        obs_params.dynamic_obstacle.zorder = 50
        obs_params.dynamic_obstacle.trajectory.draw_trajectory = False
        obs_params.lanelet_network.lanelet.show_label = False
        obs_params.lanelet_network.lanelet.left_bound_color = "#b1b1b1"
        obs_params.lanelet_network.lanelet.right_bound_color = "#b1b1b1"
        plot_limits = [-7, 30, -5, 13]
        rnd = MPRenderer(plot_limits=plot_limits, figsize=(6, 3), draw_params=obs_params)
        scenario.draw(rnd, draw_params=obs_params)

        ego_params = DynamicObstacleParams()
        ego_params.time_begin = 0
        ego_params.draw_icon = True
        ego_params.show_label = False
        ego_params.zorder = 50
        ego_params.trajectory.draw_trajectory = False
        ego_params.vehicle_shape.occupancy.shape.facecolor = "#E37222"
        ego_params.vehicle_shape.occupancy.shape.edgecolor = "#9C4100"
        ego_params.vehicle_shape.occupancy.shape.zorder = 50
        ego_params.vehicle_shape.occupancy.shape.opacity = 1
        frenet_ego = frenetix.ego_vehicle_history[-1]
        frenet_ego.draw(rnd, draw_params=ego_params)
        rnd.render()

        rnd.ax.plot(
            interface.reference_trajectory.reference_path[:, 0],
            interface.reference_trajectory.reference_path[:, 1],
            color="g",
            zorder=23,
            linewidth=1.5,
            label="reference path",
        )

        for traj in traj_states:
            rnd.ax.plot(
                traj[:, all_trajs.x_idx],
                traj[:, all_trajs.y_idx],
                color="r",
                markersize=1.5,
                zorder=21,
                linewidth=1.0,
                alpha=0.5,
            )

        valid_traj = [i for i in frenetix.all_traj if i.valid and i.feasible]
        invalid_traj = [i for i in frenetix.all_traj if not (i.valid and i.feasible)]
        keys = sorted(set([np.round(i.curvilinear.d[-1], 0) for i in frenetix.all_traj]))

        trajs_per_d_valid = dict.fromkeys(keys, None)
        trajs_per_d_invalid = dict.fromkeys(keys, None)
        for i in valid_traj:
            key = np.round(i.curvilinear.d[-1], 0)
            if trajs_per_d_valid[key] is None:
                trajs_per_d_valid[key] = []
            trajs_per_d_valid[key].append(i)
        for i in invalid_traj:
            key = np.round(i.curvilinear.d[-1], 0)
            if trajs_per_d_invalid[key] is None:
                trajs_per_d_invalid[key] = []
            trajs_per_d_invalid[key].append(i)

        step = 1
        for idx, (key, traj_list) in enumerate(trajs_per_d_valid.items()):
            if key < -2.2 or (idx + 1) % 2 == 0:
                continue
            trajs_per_d_valid[key] = sorted(traj_list, key=lambda i: i.curvilinear.s[-1], reverse=True)
            print(len(trajs_per_d_valid[key]))
            for idx in range(0, len(trajs_per_d_valid[key]), step):
                traj = trajs_per_d_valid[key][idx]
                rnd.ax.plot(
                    traj.cartesian.x,
                    traj.cartesian.y,
                    color="b" if (traj.valid and traj.feasible) else "#808080",  # "#E37222",
                    markersize=1.5,
                    zorder=21,
                    linewidth=1.0,
                    alpha=0.5,
                )

        for key, traj_list in trajs_per_d_invalid.items():
            trajs_per_d_invalid[key] = sorted(traj_list, key=lambda i: i.curvilinear.s[-1], reverse=True)
            print(len(trajs_per_d_invalid[key]))

        obs_patch = matplotlib.patches.Patch(facecolor="#2266e3", edgecolor="#003359", label="Obstacle vehicle")
        ego_patch = matplotlib.patches.Patch(facecolor="#E37222", edgecolor="#9C4100", label="Ego vehicle")
        rnd.ax.legend(
            handles=[
                ego_patch,
                obs_patch,
                matplotlib.lines.Line2D([], [], color="g", label="Global reference path"),
                matplotlib.lines.Line2D([], [], color="b", label="Analytic MPs"),
                matplotlib.lines.Line2D([], [], color="r", label="MP-RBFN"),
            ],
            handler_map={ego_patch: EgoPatchHandler(), obs_patch: ObsPatchHandler()},
            loc="upper left",
        )
        matplotlib.use("TkAgg")
        plt.pause(0.0001)

    draw(traj_states)


if __name__ == "__main__":

    models = {
        "extended_rbf_woInt_gaussian_1024_kinematic_single_track_steering_jerk_wo_acc_w_delta": "ExtendedRBF_woInt",
        "extended_rbf_gaussian_1024_kinematic_single_track_steering_jerk_wo_acc_w_delta": "ExtendedRBF",
        "mlp1layer_sig_1024_kinematic_single_track_steering_jerk_wo_acc_w_delta": "MLP1Layer",
        # "mlp2layer_sig_256_kinematic_single_track_steering_jerk_wo_acc_w_delta": "MLP2Layer",
        # "mlp1layer_tanh_1024_kinematic_single_track_steering_jerk_wo_acc_w_delta": "MLP1Layer",
    }
    test_accuracy(models)
    # test_frenetix()
    # vis_accuracy()
