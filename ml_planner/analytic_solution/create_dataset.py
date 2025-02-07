__author__ = "Marc Kaufed"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

import numpy as np
import tqdm
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from multiprocessing import Pool

from .optimal_control_planner import OCPlanner
from ml_planner.helper_functions.data_types import StateVector

warnings.filterwarnings("ignore", category=UserWarning)


def _create_partial_dataset(args):
    """Create a partial dataset for a given v0,
    sent to a worker process"""
    # read all inputs
    v0, x_interval, y_interval, psi_interval, planning_kwargs = args

    initial_state = planning_kwargs['initial_state']
    initial_state.v = float(v0)
    dt = planning_kwargs['dt']
    planning_horizon = planning_kwargs['planning_horizon']
    dynamic_model = planning_kwargs['dynamic_model']
    vehicle_model = planning_kwargs['vehicle_model']
    vehicle_params = planning_kwargs['vehicle_params']
    plot_dir = planning_kwargs['plot_dir']
    # initialize planner instance
    planner = OCPlanner(dt=dt, dynamic_model=dynamic_model, vehicle_model=vehicle_model, plot_dir=plot_dir,
                        jerk_control=planning_kwargs['jerk_control'])

    # get x-array based on reachable area
    dx = x_interval[2]
    x_min = np.round(np.ceil(v0**2/(2*vehicle_params.longitudinal.a_max)/dx)*dx, 3)
    planner.max_acc_zero_delta(initial_state, time_horizon=planning_horizon)
    x_max = np.round(np.floor(planner.xF[0]/dx)*dx, 3)
    n_x = int((x_max - x_min)/dx)
    x_array = np.linspace(x_min, x_max, n_x+1, endpoint=True)

    # get y-array
    dy = y_interval[2]
    y_min = y_interval[0]
    y_max = y_interval[1]
    n_y = int((y_max - y_min)/dy)
    y_array = np.linspace(y_min, y_max, n_y+1, endpoint=True)

    # get psi-array
    dpsi = psi_interval[2]
    psi_min = psi_interval[0]
    psi_max = psi_interval[1]
    n_psi = int((psi_max - psi_min)/dpsi)
    psi_array = np.linspace(psi_min, psi_max, n_psi+1, endpoint=True)

    data = []
    for d_xf in x_array:
        for d_yf in y_array:
            for d_psif in psi_array:
                # plan to goal state with optimal control solver
                goal_state = StateVector(x=initial_state.x+float(d_xf),
                                         y=initial_state.y+float(d_yf),
                                         delta=initial_state.delta,
                                         v=initial_state.v,
                                         psi=initial_state.psi+float(d_psif),
                                         psi_d=initial_state.psi_d,
                                         beta=initial_state.beta,
                                         alpha=initial_state.alpha
                                         )
                planner.plan_to_state(initial_state, goal_state, time_horizon=planning_horizon)

                if planner.x_mat is not None and planner.u is not None:
                    # title = f"""x0_x={np.round(planner.x0[0],1)}_
                    # y={np.round(planner.x0[1],1)}_delta={np.round(planner.x0[2],1)}_
                    # v={np.round(planner.x0[3],1)}_psi={np.round(planner.x0[4],1)}_
                    # xF_x={np.round(planner.xF[0],1)}_y={np.round(planner.xF[1],1)}_delta={np.round(planner.xF[2],1)}_
                    # v={np.round(planner.xF[3],1)}_psi={np.round(planner.xF[4],1)}"""
                    # planner.draw_trajectory(save=True, show=False, title=title)
                    x_mat = planner.x_mat
                    u = planner.u
                    data.append(np.vstack((x_mat, u)))
    data = np.array(data)
    return data


def create_dataset(v_interval, x_interval, y_interval, psi_interval,
                   planning_kwargs, data_dir, name, num_workers=1):
    """Create a dataset for a given initial velocity, final x, final y and final yaw angle interval"""
    v_max = v_interval[1]
    v_min = v_interval[0]
    dv = v_interval[2]
    n_v = int((v_max-v_min)/dv)
    v_array = np.linspace(v_min, v_max, n_v+1, endpoint=True)

    args = ((v, x_interval, y_interval, psi_interval, planning_kwargs) for v in v_array)

    data = []
    with Pool(num_workers) as p:
        # send partial dataset creation to workers
        proc_bar = tqdm.tqdm(p.imap(_create_partial_dataset, args), total=len(v_array), position=0, leave=False)
        proc_bar.set_description(f'v in [{v_min}, {v_max}]: {v_min}')
        i = 0
        for date in proc_bar:
            proc_bar.set_description(f'v in [{v_min}, {v_max}]: {v_array[i]}')

            data.append(date)
            np.savez_compressed(data_dir / (name + "_v_" + str(v_array[i]) + ".npz"), date)
            tmp_data = np.concatenate(data, axis=0)
            np.savez_compressed(data_dir / (name + "_complete_until_v_" + str(v_array[i]) + ".npz"), tmp_data)

            for psi_f in np.unique(np.round(date[:, 4, -1], 1)):
                # plot data for each psi_f and v0
                psi_data = date[np.round(date[:, 4, -1], 1) == psi_f]
                plot_dataset(psi_data,
                             path=data_dir / (name + "_v0=" + str(v_array[i]) + "_psif=" + str(psi_f) + ".svg"))
            i += 1

    data = np.concatenate(data, axis=0)

    print("finished planning")
    return data


# Plot Dataset ##################
def plot_dataset(data, path):
    cmap = mpl.colormaps["viridis_r"].resampled(len(np.unique(data[:, 0, -1])))

    segments = [np.column_stack([sample[0, :], sample[1, :]]) for sample in reversed(data)]

    # Create a color array based on the x-values
    color_idx = np.array([sample[0, -1] for sample in reversed(data)])

    # Create a LineCollection
    lc = LineCollection(segments, cmap=cmap)
    lc.set_array(color_idx)

    # Create a plot
    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.autoscale()

    # Add a colorbar
    axcb = fig.colorbar(lc)
    axcb.set_label('X Position')

    plt.savefig(path, format="svg")
    plt.close()
