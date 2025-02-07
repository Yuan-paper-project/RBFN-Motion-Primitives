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
import timeit
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from ml_planner.planner.networks.utils import DTYPE
from ml_planner.training.data_loader import DataSetAllLabels
from ml_planner.planner.planner_utils import load_model
from ml_planner.helper_functions.logging import logger_initialization

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.draw_params import MPDrawParams, DynamicObstacleParams

from frenetix_motion_planner.polynomial_trajectory import QuinticTrajectory
from frenetix_motion_planner.reactive_planner import ReactivePlannerPython
from cr_scenario_handler.utils.configuration_builder import ConfigurationBuilder
from frenetix_motion_planner.trajectories import TrajectorySample
from frenetix_motion_planner.state import ReactivePlannerState


tumblue = "#0065BD"
darkblue = "#005293"
orange = "#E37222"
green = "#A2AD00"


def vis_accuracy():
    cwd = Path.cwd()
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    device = "cpu"
    print(f"Using {device} device")

    dataset_name = "dataset_KS_BMW320i_psi_full_complete.npz"
    dataset_dir = "dataset_KS_BMW320i_psi_full"

    dataset: Callable = DataSetAllLabels

    data_dir = cwd.parent / 'dataset' / dataset_dir / dataset_name

    dataset = dataset(data_dir)

    # Test Cases
    v_idx = dataset.labels.index("v")
    x_idx = dataset.labels.index("x")
    y_idx = dataset.labels.index("y")
    psi_idx = dataset.labels.index("psi")

    y_f1 = 12
    v_0 = 10.
    x_f1 = 21
    psi_f1 = 0.16 * 10

    mask_v0 = dataset.data[:, v_idx, 0] == v_0
    data_v0 = dataset.data[mask_v0]
    mask_yf = torch.isin(data_v0[:, y_idx, -1].round(decimals=0), y_f1)
    data_yf = data_v0[mask_yf]
    mask_psif = torch.isin(data_yf[:, psi_idx, -1].round(decimals=2), psi_f1)
    data_psif = data_yf[mask_psif]
    mask_xf = torch.isin(data_psif[:, x_idx, -1].round(decimals=0), x_f1)
    gt_trajs = data_psif[mask_xf]

    y = gt_trajs[:, y_idx, -1].round().reshape(-1, 1).type(DTYPE)
    x = gt_trajs[:, x_idx, -1].round().reshape(-1, 1).type(DTYPE)
    psi = gt_trajs[:, psi_idx, -1].reshape(-1, 1).type(DTYPE)
    v = torch.tensor(v_0, dtype=DTYPE).repeat(y.shape[0], 1)

    x0_loc = torch.tensor([0.0, 0.0, 0.0], dtype=DTYPE).repeat(y.shape[0], 1)
    input_x = torch.cat([x0_loc, v, x, y, psi], dim=1).to(device)

    with torch.no_grad():
        # RBFN Model
        RBFN = "full_model_wo_acc_512rbf"
        RBFN_path = cwd / "ml_planner" / "planner" / "models" / RBFN
        RBFN_type = "ExtendedRBF"

        rbfn_model, info, dT, horizon = load_model(RBFN_path, RBFN_type, device)
        rbfn_model = rbfn_model.to(device)
        rbfn_model.eval()
        rbfn_pred = rbfn_model(input_x).cpu().detach().numpy()

        # MLP Model
        MLP = "GNN_relu_wo_acc_256"
        MLP_path = cwd / "ml_planner" / "planner" / "models" / MLP
        MLP_type = "MLP"

        mlp_model, info, dT, horizon = load_model(MLP_path, MLP_type, device)
        mlp_model = mlp_model.to(device)
        mlp_pred = mlp_model(input_x).cpu().detach().numpy()

        # Simple RBF
        SRBF = "SimpleRBF_wo_acc_512rbf"
        SRBF_path = cwd / "ml_planner" / "planner" / "models" / SRBF
        SRBF_type = "SimpleRBF"

        srbfn_model, info, dT, horizon = load_model(SRBF_path, SRBF_type, device)
        srbfn_model = srbfn_model.to(device)
        srbfn_pred = srbfn_model(input_x).cpu().detach().numpy()

        # MPRBF*
        SRBF = "RBF_wo_interp_wo_acc_512rbf"
        SRBF_path = cwd / "ml_planner" / "planner" / "models" / SRBF
        SRBF_type = "ExtendedRBF_woInt"

        rbfn2_model, info, dT, horizon = load_model(SRBF_path, SRBF_type, device)
        rbfn2_model = rbfn2_model.to(device)
        rbfn2_pred = rbfn2_model(input_x).cpu().detach().numpy()

    # draw road
    DATA_PATH = cwd / "example_scenarios"
    scenario_path = DATA_PATH / "ZAM_Tjunction-1_42_T-2.xml"
    scenario, planning_problem_set = CommonRoadFileReader(scenario_path).open()
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[1]

    # frenetix
    mod_path = Path(inspect.getfile(ReactivePlannerPython)).parent.parent
    config_sim = ConfigurationBuilder.build_sim_configuration('abc', 'abc', str(mod_path))
    config_planner = ConfigurationBuilder.build_frenetplanner_configuration('abc')
    config_planner.debug.use_cpp = False
    config_planner.debug.multiproc = False

    msg_logger = logger_initialization(log_path=str(DATA_PATH), logger="Frenetix",
                                       loglevel="DEBUG")
    frenetix = ReactivePlannerPython(config_planner, config_sim, scenario, planning_problem,
                                     str(DATA_PATH), mod_path, msg_logger)

    x_ref = np.linspace(-10, 160, 340)
    y_ref = np.zeros_like(x_ref)
    ref_path = np.vstack((x_ref, y_ref)).T
    frenetix.set_reference_and_coordinate_system(ref_path)

    s0, d0 = frenetix.coordinate_system.convert_to_curvilinear_coords(0, 0)
    x0_long = np.array([s0, v_0, 0])
    x_d_long = np.array([s0+x_f1, gt_trajs[:, v_idx, -1]*np.cos(1.57), 0])
    traj_long = QuinticTrajectory(tau_0=0, delta_tau=3, x_0=x0_long, x_d=x_d_long)

    x0_lat = np.array([d0, 0, 0])
    x_d_lat = np.array([y_f1, gt_trajs[:, v_idx, -1]*np.sin(1.57), 0])
    traj_lat = QuinticTrajectory(tau_0=0, delta_tau=3, x_0=x0_lat, x_d=x_d_lat)
    trajectory = TrajectorySample(frenetix.horizon, frenetix.dT,
                                  traj_long, traj_lat, 0, frenetix.cost_function.cost_weights)
    traj_sample = frenetix.check_feasibility([trajectory])
    traj_fren1 = traj_sample[0].cartesian

    # VISUALIZE
    visualize_scenario(dataset, scenario, gt_trajs, rbfn_pred, rbfn2_pred, mlp_pred, srbfn_pred,
                       traj_fren1)

    visualize_velocity(dataset, gt_trajs, rbfn_pred, rbfn2_pred, mlp_pred, srbfn_pred, traj_fren1)


def visualize_scenario(dataset, scenario, gt_trajs, rbfn_pred, rbfn2_pred, mlp_pred, srbfn_pred,
                       traj_fren):
    x_idx = dataset.labels.index("x")
    y_idx = dataset.labels.index("y")
    obs_params = MPDrawParams()
    obs_params.dynamic_obstacle.time_begin = 0
    obs_params.dynamic_obstacle.draw_icon = True
    obs_params.dynamic_obstacle.show_label = True
    obs_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#2266e3"
    obs_params.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = "#003359"
    obs_params.axis_visible = True
    obs_params.lanelet_network.lanelet.show_label = False
    obs_params.lanelet_network.lanelet.left_bound_color = "#b1b1b1"
    obs_params.lanelet_network.lanelet.right_bound_color = "#b1b1b1"

    plot_limits = [-6, 55, -3, 13]
    rnd = MPRenderer(plot_limits=plot_limits,
                     figsize=(6, 3),  draw_params=obs_params)
    scenario.lanelet_network.draw(rnd, draw_params=obs_params)

    ego_params = DynamicObstacleParams()
    ego_params.time_begin = 0
    ego_params.draw_icon = True
    ego_params.show_label = False
    ego_params.vehicle_shape.occupancy.shape.facecolor = "#E37222"
    ego_params.vehicle_shape.occupancy.shape.edgecolor = "#9C4100"
    ego_params.vehicle_shape.occupancy.shape.zorder = 50
    ego_params.vehicle_shape.occupancy.shape.opacity = 1
    ego_params.trajectory.draw_trajectory = False
    scenario.dynamic_obstacles[0].draw(rnd, draw_params=ego_params)
    rnd.render()

    rnd.ax.plot(gt_trajs[:, x_idx, :].T-2, gt_trajs[:, y_idx, :].T,
                linestyle="dashed", color='k', zorder=21, label="OCP")

    rnd.ax.plot(traj_fren.x-2, traj_fren.y, color='purple', zorder=21, label="Semi-Analytic MP")
    rnd.ax.plot(rbfn2_pred[0, 0, :].T-2, rbfn2_pred[0, 1, :].T, color=tumblue, zorder=21, label="MP-RBFN (no interp)")
    rnd.ax.plot(mlp_pred[0, 0, :].T-2, mlp_pred[0, 1, :].T, color=green, zorder=21, label="GNN")
    rnd.ax.plot(rbfn_pred[0, 0, :].T-2, rbfn_pred[0, 1, :].T, color=orange, zorder=21, label="MP-RBFN")
    rnd.ax.plot(srbfn_pred[0, 0, :].T-2, srbfn_pred[0, 1, :].T, color='red', zorder=21, label="Simple RBFN")
    rnd.ax.legend()
    matplotlib.use("TkAgg")
    plt.show()


def visualize_velocity(dataset, gt_trajs, rbfn_pred, rbfn2_pred, mlp_pred, srbfn_pred, traj_fren):
    def arrowed_spines(fig, ax):

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        xmin -= 1
        xmax += 1
        ymin -= 1
        ymax += 1
        # removing the default axis on all sides:
        # for side in ['bottom','right','top','left']:
        #     ax.spines[side].set_visible(False)

        # removing the axis ticks
        plt.xticks([])  # labels
        plt.yticks([])
        ax.xaxis.set_ticks_position('none')  # tick markers
        ax.yaxis.set_ticks_position('none')

        # get width and height of axes object to compute
        # matching arrowhead length and width
        dps = fig.dpi_scale_trans.inverted()
        bbox = ax.get_window_extent().transformed(dps)
        width, height = bbox.width, bbox.height

        # manual arrowhead width and length
        hw = 1./20.*(ymax-ymin)
        hl = 1./20.*(xmax-xmin)
        lw = 1.  # axis line width
        ohg = 0.3  # arrow overhang

        # compute matching arrowhead length and width
        yhw = hw/(ymax-ymin)*(xmax-xmin) * height/width
        yhl = hl/(xmax-xmin)*(ymax-ymin) * width/height

        # draw x and y axis
        ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw=lw,
                 head_width=hw, head_length=hl, overhang=ohg,
                 length_includes_head=True, clip_on=False)

        ax.arrow(0, ymin, 0., ymax-ymin, fc='k', ec='k', lw=lw,
                 head_width=yhw, head_length=yhl, overhang=ohg,
                 length_includes_head=True, clip_on=False)

    v_idx = dataset.labels.index("v")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(gt_trajs[:, v_idx, :].T, linestyle="dashed", color='k', zorder=21, label="OCP")
    ax.plot(traj_fren.v, color='purple', zorder=21, label="Semi-Analytic MP")
    ax.plot(rbfn2_pred[0, 2, :].T, color=tumblue, zorder=21, label="MP-RBFN")
    ax.plot(mlp_pred[0, 2, :].T, color=green, zorder=21)
    ax.plot(srbfn_pred[0, 2, :].T, color='red', zorder=21, label="Simple RBFN")
    ax.plot(rbfn_pred[0, 2, :].T, color=orange, zorder=21, label="MP-RBFN")
    ax.plot(rbfn_pred[1, 2, :].T, color=orange, zorder=21)
    ax.set_xlabel("Time")
    ax.set_ylabel("Velocity in m/s")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    arrowed_spines(fig, ax)
    plt.show()

    # Save the DataFrame to a CSV file
    data = np.vstack((gt_trajs[:, v_idx, :], traj_fren.v.reshape(1, -1), rbfn2_pred[0, 2, :],
                      mlp_pred[0, 2, :], rbfn_pred[0, 2, :]))
    df = pd.DataFrame(data)
    df.to_csv('vel_data.csv', index=False)


def test_accuracy():
    cwd = Path.cwd()
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    # device = "cpu"
    print(f"Using {device} device")

    dataset_name = "dataset_KS_BMW320i_psi_full_complete.npz"
    dataset_dir = "dataset_KS_BMW320i_psi_full"

    dataset: Callable = DataSetAllLabels

    data_dir = cwd.parent / 'dataset' / dataset_dir / dataset_name

    dataset = dataset(data_dir)

    # Test Cases
    v_idx = dataset.labels.index("v")
    x_idx = dataset.labels.index("x")
    y_idx = dataset.labels.index("y")
    psi_idx = dataset.labels.index("psi")

    psi = True
    if psi:
        values = np.linspace(-1.6, 1.6, 21, endpoint=True)
    else:
        values = [50, 180, 800, 3500, 13000, 90000]
    for j in values:
        if psi:
            mask = dataset.data[:, psi_idx, -1].round(decimals=2) == round(j, 2)
            data = dataset.data[mask]
        else:
            data = dataset.data[:j]
        if data.shape[0] == 0:
            raise ValueError("No data for psi_f")
        gt_trajs = data.cpu().numpy()

        y = data[:, y_idx, -1].round().reshape(-1, 1).type(DTYPE)
        x = data[:, x_idx, -1].round().reshape(-1, 1).type(DTYPE)
        psi = data[:, psi_idx, -1].reshape(-1, 1).type(DTYPE)
        v = data[:, v_idx, 0].reshape(-1, 1).type(DTYPE)

        x0_loc = torch.tensor([0.0, 0.0, 0.0], dtype=DTYPE).repeat(y.shape[0], 1)
        input_x = torch.cat([x0_loc, v, x, y, psi], dim=1).to(device)

        with torch.no_grad():
            # RBFN Model
            RBFN = "full_model_wo_acc_512rbf"
            RBFN_path = cwd / "ml_planner" / "planner" / "models" / RBFN
            RBFN_type = "ExtendedRBF"

            rbfn_model, info, dT, horizon = load_model(RBFN_path, RBFN_type, device)

            rbfn_model = rbfn_model.to(device)
            rbfn_model.eval()
            rbfn_pred = rbfn_model(input_x).cpu().detach().numpy()
            rbfn_pos = np.sqrt((rbfn_pred[:, 0, :] - gt_trajs[:, 0, :])**2
                               + (rbfn_pred[:, 1, :] - gt_trajs[:, 1, :])**2)
            rnfn_vel = np.sqrt((rbfn_pred[:, 2, :] - gt_trajs[:, 3, :])**2)
            rbfn_or = np.sqrt((rbfn_pred[:, 3, :] - gt_trajs[:, 4, :])**2)
            num_rep = 1000
            rbfn_time = []
            for i in range(num_rep):
                rbfn_time.append(timeit.timeit(lambda: rbfn_model(input_x), number=1))

            # MLP Model
            MLP = "GNN_relu_wo_acc_256"
            MLP_path = cwd / "ml_planner" / "planner" / "models" / MLP
            MLP_type = "MLP"

            mlp_model, info, dT, horizon = load_model(MLP_path, MLP_type, device)
            mlp_model = mlp_model.to(device)
            mlp_pred = mlp_model(input_x).cpu().detach().numpy()
            mlp_pos = np.sqrt((mlp_pred[:, 0, :] - gt_trajs[:, 0, :])**2 +
                              (mlp_pred[:, 1, :] - gt_trajs[:, 1, :])**2)
            mlp_vel = np.sqrt((mlp_pred[:, 2, :] - gt_trajs[:, 3, :])**2)
            mlp_or = np.sqrt((mlp_pred[:, 3, :] - gt_trajs[:, 4, :])**2)

            mlp_time = []
            for i in range(num_rep):
                mlp_time.append(timeit.timeit(lambda: mlp_model(input_x), number=1))

            # Simple RBF
            SRBF = "SimpleRBF_wo_acc_512rbf"
            SRBF_path = cwd / "ml_planner" / "planner" / "models" / SRBF
            SRBF_type = "SimpleRBF"

            srbfn_model, info, dT, horizon = load_model(SRBF_path, SRBF_type, device)
            srbfn_model = srbfn_model.to(device)
            srbfn_pred = srbfn_model(input_x).cpu().detach().numpy()
            srbfn_pos = np.sqrt((srbfn_pred[:, 0, :] - gt_trajs[:, 0, :])**2 +
                                (srbfn_pred[:, 1, :] - gt_trajs[:, 1, :])**2)
            srbfn_vel = np.sqrt((srbfn_pred[:, 2, :] - gt_trajs[:, 3, :])**2)
            srbfn_or = np.sqrt((srbfn_pred[:, 3, :] - gt_trajs[:, 4, :])**2)

            # MPRBF without interpolation
            SRBF = "RBF_wo_interp_wo_acc_512rbf"
            SRBF_path = cwd / "ml_planner" / "planner" / "models" / SRBF
            SRBF_type = "ExtendedRBF_woInt"

            rbfn2_model, info, dT, horizon = load_model(SRBF_path, SRBF_type, device)
            rbfn2_model = rbfn2_model.to(device)
            rbfn2_pred = rbfn2_model(input_x).cpu().detach().numpy()
            rbfn2_pos = np.sqrt((rbfn2_pred[:, 0, :] - gt_trajs[:, 0, :])**2 +
                                (rbfn2_pred[:, 1, :] - gt_trajs[:, 1, :])**2)
            rnfn2_vel = np.sqrt((rbfn2_pred[:, 2, :] - gt_trajs[:, 3, :])**2)
            rbfn2_or = np.sqrt((rbfn2_pred[:, 3, :] - gt_trajs[:, 4, :])**2)
            rbfn2_time = []
            for i in range(num_rep):
                rbfn2_time.append(timeit.timeit(lambda: rbfn2_model(input_x), number=1))

        print(f"{j}")
        print(f"MP-RBFN Position Error: Mean {rbfn_pos.mean()}, Max: {rbfn_pos.max()}")
        print(f"MP-RBFN Velocity Error: Mean {rnfn_vel.mean()}, Max: {rnfn_vel.max()}")
        print(f"MP-RBFN Orientation Error: Mean {rbfn_or.mean()}, Max: {rbfn_or.max()}")
        print(f"MP-RBFN* Position Error: Mean {rbfn2_pos.mean()}, Max: {rbfn2_pos.max()}")
        print(f"MP-RBFN* Velocity Error: Mean {rnfn2_vel.mean()}, Max: {rnfn2_vel.max()}")
        print(f"MP-RBFN* Orientation Error: Mean {rbfn2_or.mean()}, Max: {rbfn2_or.max()}")
        print(f"GNN Position Error: Mean {mlp_pos.mean()}, Max: {mlp_pos.max()}")
        print(f"GNN Velocity Error: Mean {mlp_vel.mean()}, Max: {mlp_vel.max()}")
        print(f"GNN Orientation Error: Mean {mlp_or.mean()}, Max: {mlp_or.max()}")
        print(f"Simple RBFN Position Error: Mean {srbfn_pos.mean()}, Max: {srbfn_pos.max()}")
        print(f"Simple RBFN Velocity Error: Mean {srbfn_vel.mean()}, Max: {srbfn_vel.max()}")
        print(f"Simple RBFN Orientation Error: Mean {srbfn_or.mean()}, Max: {srbfn_or.max()}")
        print(j)
        rbfn_time = np.array(rbfn_time)
        mlp_time = np.array(mlp_time)
        rbfn2_time = np.array(rbfn2_time)
        print(f"MP-RBFN Time: MEan {rbfn_time.mean()*1e3}, Max: {rbfn_time.max()*1e3}")
        print(f"MP-RBFN* Time: Mean {rbfn2_time.mean()*1e3}, Max: {rbfn2_time.max()*1e3}")
        print(f"GNN Time: Mean {mlp_time.mean()*1e3}, Max: {mlp_time.max()*1e3}")


def test_frenetix():
    cwd = Path.cwd()
    dataset_name = "dataset_KS_BMW320i_psi_full_complete.npz"
    dataset_dir = "dataset_KS_BMW320i_psi_full"

    dataset: Callable = DataSetAllLabels

    data_dir = cwd.parent / 'dataset' / dataset_dir / dataset_name

    dataset = dataset(data_dir)

    DATA_PATH = cwd / "example_scenarios"
    scenario_path = DATA_PATH / "ZAM_Tjunction-1_42_T-2.xml"
    scenario, planning_problem_set = CommonRoadFileReader(scenario_path).open()
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

    # Test Cases
    psi_idx = dataset.labels.index("psi")
    # frenetix
    mod_path = Path(inspect.getfile(ReactivePlannerPython)).parent.parent
    config_sim = ConfigurationBuilder.build_sim_configuration('abc', 'abc', str(mod_path))
    config_planner = ConfigurationBuilder.build_frenetplanner_configuration('abc')
    config_planner.debug.use_cpp = False
    config_planner.debug.multiproc = False

    msg_logger = logger_initialization(log_path=str(DATA_PATH), logger="Frenetix",
                                       loglevel="DEBUG")
    frenetix = ReactivePlannerPython(config_planner, config_sim, scenario, planning_problem,
                                     str(DATA_PATH), mod_path, msg_logger)
    x_ref = np.linspace(-10, 160, 340)
    y_ref = np.zeros_like(x_ref)
    ref_path = np.vstack((x_ref, y_ref)).T
    frenetix.set_reference_and_coordinate_system(ref_path)

    for psi_f in np.linspace(-1.6, 1.6, 21, endpoint=True):

        mask = dataset.data[:, psi_idx, -1].round(decimals=2) == round(psi_f, 2)
        data = dataset.data[mask].numpy()
        if data.shape[0] == 0:
            raise ValueError("No data for psi_f")
        trajs = []
        pos_errors = []
        vel_errors = []
        or_errors = []
        trajs_valid = 0
        for i in range(data.shape[0]):
            s0, d0 = frenetix.coordinate_system.convert_to_curvilinear_coords(0, 0)
            sd, dd = frenetix.coordinate_system.convert_to_curvilinear_coords(data[i, 0, -1], data[i, 1, -1])
            v0 = data[i, 3, 0]
            vd = data[i, 3, -1]
            x0_long = np.array([s0, v0, 0])
            x_d_long = np.array([sd, vd, 0])
            frenetix.x_0 = ReactivePlannerState(0, np.array([0, 0]), 0, v0, 0, 0)
            traj_long = QuinticTrajectory(tau_0=0, delta_tau=3, x_0=x0_long, x_d=x_d_long)

            x0_lat = np.array([d0, 0, 0])
            x_d_lat = np.array([dd, 0, 0])
            traj_lat = QuinticTrajectory(tau_0=0, delta_tau=3, x_0=x0_lat, x_d=x_d_lat)
            trajectory = TrajectorySample(frenetix.horizon, frenetix.dT, traj_long,
                                          traj_lat, 0, frenetix.cost_function.cost_weights)
            traj = frenetix.check_feasibility([trajectory])
            if traj[0].valid and traj[0].feasible:
                cart = traj[0].cartesian
                dx = data[i, 0, :] - cart.x
                dy = data[i, 1, :] - cart.y
                pos_errors.append(np.sqrt(dx**2 + dy**2).mean())
                vel_errors.append(np.sqrt((data[i, 3, :] - cart.v)**2).mean())
                or_errors.append(np.sqrt((data[i, 4, :] - cart.theta)**2).mean())
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


if __name__ == "__main__":
    test_accuracy()
    test_frenetix()
    vis_accuracy()
