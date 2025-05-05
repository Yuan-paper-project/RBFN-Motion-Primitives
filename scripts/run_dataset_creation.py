__author__ = "Marc Kaufed"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

import os
from pathlib import Path
import shutil
import yaml
import itertools
import time
import numpy as np
import tqdm
from scipy.optimize import fsolve

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from multiprocessing import Pool
from functools import partial

from ml_planner.general_utils.logging import logger_initialization
from ml_planner.analytic_solution.ocp import main as solve_ocp
from ml_planner.general_utils.vehicle_models import VEHICLE_PARAMS, STATE_LABELS, CONTROL_LABELS

# from ml_planner.vehicle_models.acceleration_constraints import acceleration_constraints

###############################
# PATH AND DEBUG CONFIGURATION
CWD = Path.cwd()
DATA_PATH = CWD.parent / "dataset"
LOG_PATH = CWD / "logs"

CONFIG_FILE = Path(__file__).parent / "config.yaml"

# debug configurations
DATASET_NAME_ENDING = "v2"

DELETE_ALL_FORMER_LOGS = False
LOGGING_LEVEL = "Info"

NUM_PROCESSES = 1
SOLVER = "heun"  # "rk4", "heun", "euler"
################################


def plot_dataset(data_unfiltered, state_labels, path):
    if "delta" in state_labels:
        delta_idx = state_labels.index("delta") if "delta" in state_labels else None
        delta_array = np.unique(np.round(data_unfiltered[:, 0, delta_idx], 2))
    else:
        delta_array = range(1)
    v_idx = state_labels.index("v")
    x_idx = state_labels.index("x")
    y_idx = state_labels.index("y")
    psi_idx = state_labels.index("psi")

    for delta in delta_array:
        if "delta" in state_labels:
            data = data_unfiltered[np.round(data_unfiltered[:, 0, delta_idx], 2) == delta]
        else:
            data = data_unfiltered
        for psi in np.unique(
            np.round(
                data[
                    :,
                    -1,
                    psi_idx,
                ],
                2,
            )
        ):
            dat = data[
                np.round(
                    data[
                        :,
                        -1,
                        psi_idx,
                    ],
                    2,
                )
                == psi
            ]

            cmap = mpl.colormaps["viridis_r"].resampled(len(np.unique(dat[:, -1, x_idx])))

            segments = [np.column_stack([sample[:, x_idx], sample[:, y_idx]]) for sample in reversed(dat)]

            # Create a color array based on the x-values
            color_idx = np.array([sample[-1, x_idx] for sample in reversed(dat)])

            # Create a LineCollection
            lc = LineCollection(segments, cmap=cmap)
            lc.set_array(color_idx)

            # Create a plot
            fig, ax = plt.subplots()
            ax.add_collection(lc)
            ax.autoscale()
            ax.set_ylim(np.min(dat[..., 1]) - 1, np.max(dat[..., 1]) + 1)
            # Add a colorbar
            axcb = fig.colorbar(lc)
            axcb.set_label("X Position")
            name = f"v0={dat[0, 0, v_idx]}_psif={psi}_delta={delta}.svg"
            plt.savefig(path / name, format="svg")
            plt.close()


def run_dataset_creation():
    start_time = time.time()
    if DELETE_ALL_FORMER_LOGS:
        shutil.rmtree(LOG_PATH, ignore_errors=True)
    # logging
    msg_logger = logger_initialization(
        LOG_PATH, "Dataset_Creation", loglevel=LOGGING_LEVEL.upper(), loglevel_msg="WARNING"
    )
    with open(CONFIG_FILE, "r") as file:
        config = yaml.safe_load(file)

    vehicle_config = config["vehicle"]
    vehicle_params = VEHICLE_PARAMS[vehicle_config["vehicle_type"]]
    vehicle_config["vehicle_params"] = vehicle_params

    state_labels = STATE_LABELS[vehicle_config["dynamic_model"]]
    control_labels = CONTROL_LABELS[vehicle_config["control_mode"]]

    # Directory Configuration ##################
    dataset_string = f"{vehicle_config['vehicle_type']}_{vehicle_config['dynamic_model']}_{vehicle_config['control_mode']}_{DATASET_NAME_ENDING}"
    # data directory
    data_dir = DATA_PATH / f"dataset_{dataset_string}"
    plot_dir = data_dir / "plots"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    msg_logger.warning(f"Storing data in {data_dir}")

    msg_logger.warning(f"vehicle_type: {vehicle_config['vehicle_type']}")
    msg_logger.warning(f"dynamic_model: {vehicle_config['dynamic_model']}")
    msg_logger.warning(f"control_mode: {vehicle_config['control_mode']}")
    msg_logger.info(f"state labels: {state_labels}")
    msg_logger.info(f"control labels: {control_labels}")
    # Dataset Configuration ##################
    planning_horizon = config["time"]["planning_horizon"]
    dt = config["time"]["dt"]
    n_steps = int(planning_horizon / dt) + 1

    sampling_config = config["sampling"]
    # initial velocity
    v0_min = sampling_config["v0"][0]
    v0_max = sampling_config["v0"][1]
    dv = sampling_config["dv"]
    n_v = int((v0_max - v0_min) / dv) + 1
    v0_array = np.linspace(v0_min, v0_max, n_v, endpoint=True)
    v0_array[v0_array - 0 < 0.1] = 0.1  # avoid division by zero

    # adapt vehicle parameters
    vehicle_params.longitudinal.v_max = v0_array[-1]  # set maximum velocity for vehicle model
    vehicle_params.a_lat = 0.5 * 9.81  # set maximum lateral acceleration for vehicle model

    # initial steering angle
    delta0_min = sampling_config["delta0"][0]
    delta0_max = sampling_config["delta0"][1]  # always needed for prefiltering sampling points
    if vehicle_config["control_mode"] == "steering_jerk":

        d_delta0 = sampling_config["ddelta"]
        n_delta0 = int((delta0_max - delta0_min) / d_delta0) + 1
        delta0_array = np.linspace(delta0_min, delta0_max, n_delta0, endpoint=True)
    else:
        delta0_array = np.array([0.0])

    # final heading angle
    psif_min = sampling_config["psif"][0]
    psif_max = sampling_config["psif"][1]
    d_psif = sampling_config["dpsi"]
    n_psif = int((psif_max - psif_min) / d_psif) + 1
    psif_array = np.linspace(psif_min, psif_max, n_psif, endpoint=True)

    # final position
    dx = sampling_config["dx"]
    x_min = v0_array**2 / (2 * vehicle_params.longitudinal.a_max)  # minimum stopping distance
    x_min = np.ceil(x_min / dx) * dx

    # get final position based on maximum feasible acceleration
    def forward_pos(v0_array):
        M = len(v0_array)
        # N = int(T / dt)
        # t = np.linspace(0, T, N)

        x = np.zeros((n_steps, M))
        v = np.zeros((n_steps, M))

        v[0, :] = v0_array

        for i in range(1, n_steps):

            def acc(v):
                fac = vehicle_params.longitudinal.v_switch / v
                fac[fac > 1] = 1
                fac[v >= vehicle_params.longitudinal.v_max] = 0
                a = vehicle_params.longitudinal.a_max * fac
                return a

            # euler method
            v[i, :] = v[i - 1, :] + acc(v[i - 1]) * dt
            x[i, :] = x[i - 1, :] + v[i - 1, :] * dt

            # RK4 method
            # vi = v[i - 1, :]
            # xi = x[i - 1, :]

            # # RK4 for v
            # k1v = acc(vi)
            # k2v = acc(vi + 0.5 * dt * k1v)
            # k3v = acc(vi + 0.5 * dt * k2v)
            # k4v = acc(vi + dt * k3v)
            # v[i, :] = vi + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)

            # # RK4 for x
            # k1x = vi
            # k2x = vi + 0.5 * dt * k1v
            # k3x = vi + 0.5 * dt * k2v
            # k4x = vi + dt * k3v
            # x[i, :] = xi + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)

        return x, v

    s, _ = forward_pos(v0_array)
    x_max = np.floor(s[-1, :] / dx) * dx
    n_x = ((x_max - x_min) / dx + 1).astype(int)

    dy = sampling_config["dy"]
    n_y = ((x_max - 0) / dy + 1).astype(int)

    # filter out unreachable points
    def calculate_arc_length(xf, yf, x0=0.0, y0=0.0, theta0=0.0):
        if yf == y0:
            return yf, np.inf, 0
        # Unit vector perpendicular to the initial orientation
        n = np.array([-np.sin(theta0), np.cos(theta0)])

        # Function to solve for r
        def find_radius(r):
            xc, yc = x0 + r * n[0], y0 + r * n[1]
            return (xf - xc) ** 2 + (yf - yc) ** 2 - r**2

        # Solve for r using a numerical solver
        r_initial_guess = 1.0  # Initial guess for the radius
        r = fsolve(find_radius, r_initial_guess)[0]

        # Calculate the center of the circle
        xc, yc = x0 + r * n[0], y0 + r * n[1]

        # Vectors from the center to the initial and final points
        v1 = np.array([x0 - xc, y0 - yc])
        v2 = np.array([xf - xc, yf - yc])

        # Calculate the angle subtended by the arc
        theta = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        # Calculate the arc length
        arc_length = r * theta
        return arc_length, r, theta

    pos_array = []
    for i in range(len(v0_array)):
        arr_x = np.linspace(x_min[i], x_max[i], n_x[i], endpoint=True)
        arr_y = np.linspace(0, x_max[i], n_y[i], endpoint=True)
        comb = np.array(np.meshgrid(arr_x, arr_y)).T.reshape(-1, 2)
        mask = []
        dist = np.sqrt(comb[:, 0] ** 2 + comb[:, 1] ** 2)

        R = (vehicle_params.a + vehicle_params.b) / np.tan(delta0_max)

        for j in range(len(comb)):
            arc, r, theta = calculate_arc_length(comb[j][0], comb[j][1])

            mask.append(r >= R if comb[j][1] <= R else comb[j][0] >= R)
        mask = np.logical_and(dist <= arr_x[-1], mask)
        pos_array.append(comb[mask])

    v0posf_array = np.vstack([np.hstack([np.full((j.shape[0], 1), i), j]) for i, j in zip(v0_array, pos_array)])

    combinations = list(itertools.product(v0posf_array, psif_array, delta0_array))
    msg_logger.warning(f"Number of trajectories to optimize: {len(combinations)}")

    state_data = []
    control_data = []
    run_times = []
    cur_v0 = v0_array[0]
    cur_idx = 0
    proc_bar = tqdm.tqdm(combinations)

    y_idx = state_labels.index("y")
    v_idx = state_labels.index("v")
    psi_idx = state_labels.index("psi")
    delta_idx = state_labels.index("delta") if "delta" in state_labels else None
    beta_idx = state_labels.index("beta") if "beta" in state_labels else None

    mirror_idxs = [i for i in [y_idx, psi_idx, delta_idx, beta_idx] if i is not None]
    ctr_idx = control_labels.index([i for i in control_labels if i != "j_long"][0])
    if NUM_PROCESSES <= 1:
        for pts in proc_bar:
            proc_bar.set_description(f"current v: {cur_v0}")
            v0posf = pts[0]
            delta0 = pts[2]
            psif = pts[1]

            # create dataset
            planning_config = {
                "T": planning_horizon,
                "dt": dt,
                "x0": 0,
                "y0": 0,
                "psi0": 0,
                "v0": v0posf[0],  # to be looped
                "delta0": delta0,  # to be looped
                # "beta0": 0,
                "xf": v0posf[1],  # to be looped
                "yf": v0posf[2],  # to be looped
                "psif": psif,  # to be looped
                # "psi_dotf": 0,
                "a_longf": 0,
                "deltaf": 0,
                "betaf": 0,
            }

            # run ocp
            x_vec, u_vec, run_time = solve_ocp(
                planning_config, vehicle_config, msg_logger, ivp_solver=SOLVER, draw=False
            )

            if x_vec is not None:
                if len(state_data) > 0 and abs(x_vec[0, 3] - cur_v0) >= dv:
                    # if new velocity is reached save current data
                    data_vec = np.array(state_data)
                    control_vec = np.array(control_data)

                    state_mirror = data_vec.copy()
                    state_mirror[..., mirror_idxs] *= -1

                    control_mirror = control_vec.copy()
                    control_mirror[..., ctr_idx] *= -1

                    data_vec = np.unique(np.concatenate((data_vec, state_mirror), axis=0), axis=0)
                    control_vec = np.unique(np.concatenate((control_vec, control_mirror), axis=0), axis=0)
                    plot_dataset(data_vec[cur_idx:], state_labels, plot_dir)
                    perf_data = np.array(run_times)
                    np.savez_compressed(
                        data_dir / "data.npz", state_vec=data_vec, control_vec=control_vec, perf_data=perf_data
                    )
                    cur_v0 = x_vec[0, v_idx]
                    cur_idx = len(state_data)
                state_data.append(x_vec)
                control_data.append(u_vec)
                run_times.append(run_time)
    else:
        kwargs = [
            {
                "T": planning_horizon,
                "dt": dt,
                "x0": 0,
                "y0": 0,
                "psi0": 0,
                "v0": pts[0][0],  # to be looped
                "delta0": pts[2],  # to be looped
                "xf": pts[0][1],  # to be looped
                "yf": pts[0][2],  # to be looped
                "psif": pts[1],  # to be looped
                "psi_dotf": 0,
                "a_longf": 0,
                "deltaf": 0,
                "betaf": 0,
            }
            for pts in combinations
        ]
        with Pool(processes=NUM_PROCESSES) as pool:
            func = partial(
                solve_ocp, vehicle_config=vehicle_config, msg_logger=msg_logger, ivp_solver=SOLVER, draw=False
            )
            proc_bar = tqdm.tqdm(
                pool.imap(func, kwargs, chunksize=50),
                total=len(kwargs),
                position=0,
                leave=False,
            )
            for x_vec, u_vec, run_time in proc_bar:
                if x_vec is not None:
                    state_data.append(x_vec)
                    control_data.append(u_vec)
                    run_times.append(run_time)
                    proc_bar.set_description(f"current v: {x_vec[0, v_idx]}")
                    if len(state_data) % 300 == 0:
                        data_vec = np.array(state_data)
                        control_vec = np.array(control_data)

                        state_mirror = data_vec.copy()
                        state_mirror[..., mirror_idxs] *= -1

                        control_mirror = control_vec.copy()
                        control_mirror[..., ctr_idx] *= -1
                        perf_data = np.array(run_times)
                        data_vec = np.unique(np.concatenate((data_vec, state_mirror), axis=0), axis=0)
                        control_vec = np.unique(np.concatenate((control_vec, control_mirror), axis=0), axis=0)
                        np.savez_compressed(
                            data_dir / "data.npz", state_vec=data_vec, control_vec=control_vec, perf_data=perf_data
                        )

    msg_logger.warning(f"Successfull optimization of {len(state_data)} trajectories")
    perf_data = np.array(run_times)
    msg_logger.warning(f"Average runtime ('main','solve_ocp', 't_proc', 't_wall'): {np.mean(perf_data, axis=0)} s")
    # Mirror the data since it's only calculated for one side
    data_vec = np.array(state_data)
    control_vec = np.array(control_data)

    state_mirror = data_vec.copy()
    state_mirror[..., mirror_idxs] *= -1

    control_mirror = control_vec.copy()

    control_mirror[..., ctr_idx] *= -1

    data_vec = np.unique(np.concatenate((data_vec, state_mirror), axis=0), axis=0)
    control_vec = np.unique(np.concatenate((control_vec, control_mirror), axis=0), axis=0)

    np.savez_compressed(data_dir / "data.npz", state_vec=data_vec, control_vec=control_vec, perf_data=perf_data)
    calc_time = time.time() - start_time
    msg_logger.warning(f"Dataset creation took {calc_time:.2f} seconds")

    # plot dataset
    v_list = np.unique(np.round(data_vec[:, 0, v_idx], 2))
    for v in v_list:
        data_v = data_vec[np.round(data_vec[:, 0, v_idx], 2) == v]
        if len(data_v) > 0:
            plot_dataset(data_v, state_labels, plot_dir)

    msg_logger.warning("Finished dataset creation")


if __name__ == "__main__":
    run_dataset_creation()
