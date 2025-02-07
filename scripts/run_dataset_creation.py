__author__ = "Marc Kaufed"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

import os
import numpy as np
from pathlib import Path
import warnings
import logging
from vehiclemodels.vehicle_parameters import setup_vehicle_parameters

from ml_planner.analytic_solution.utils import get_model_state_vector

from ml_planner.helper_functions.data_types import StateVector, DynamicModel, VehicleModel
from ml_planner.analytic_solution.create_dataset import create_dataset

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    """Main Script to create a dataset for the OCP training"""
    # Dynamic and Vehicle Model ##################
    dynamic_model = DynamicModel.KS
    vehicle_model = VehicleModel.BMW320i
    vehicle_params = setup_vehicle_parameters(vehicle_model.value)

    # Dataset Configuration ##################
    dataset_string = f'dataset_{dynamic_model.name}_{vehicle_model.name}_psi_full'
    # data directory
    cwd = Path.cwd()
    data_dir = cwd.parent / 'dataset' / dataset_string
    print(data_dir)
    plot_dir = data_dir / 'single_trajs'
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # logging
    os.makedirs(cwd / 'logs', exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename="logs/create_dataset.log", filemode='w', force=True)

    # define number of workers for multiprocessing
    num_workers = 28

    # Planning Configuration ##################
    planning_horizon = 3  # planning horizon in seconds

    dt = 0.1        # time step size
    x0 = 0.         # initial x position
    y0 = 0.         # initialy position
    delta0 = 0      # initial steering angle
    vel0 = 0        # initial velocity
    psi0 = 0        # initial yaw angle
    psi_d0 = 0.     # initial yaw rate (if used in dynamic model)
    beta0 = 0.      # sideslip angle at vehicle center (if used in dynamic model)
    alpha0 = 0.     # hitch angle if dynamic modle 'ks with trailer' is used

    initial_state0 = StateVector(x=x0,
                                 y=y0,
                                 delta=delta0,
                                 v=vel0,
                                 psi=psi0,
                                 psi_d=psi_d0,
                                 beta=beta0,
                                 alpha=alpha0
                                 )
    _, intial_state0 = get_model_state_vector(state_vector=initial_state0,
                                              dynamic_model=dynamic_model,
                                              vehicle_params=vehicle_params)

    # Initial velocity range ################
    dv = 1
    v_min = 0
    v_max = np.floor(vehicle_params.longitudinal.v_max)
    v_interval = [v_min, v_max, dv]

    # Goal State Position #####################
    dx = 3
    dy = 1
    # x position depends on initial velocity
    x_interval = [None, None, dx]

    y_min = -15
    y_max = 15
    y_interval = [y_min, y_max, dy]

    # Goal State Orientation ##################
    dpsi = 0.16
    psi_min = -1.6
    psi_max = 1.6
    psi_interval = [psi_min, psi_max, dpsi]

    # Create Dataset ##################
    planning_kwargs = {'initial_state': initial_state0, 'dt': dt, 'planning_horizon': planning_horizon,
                       'dynamic_model': dynamic_model, 'vehicle_model': vehicle_model, 'vehicle_params': vehicle_params,
                       'plot_dir': plot_dir, 'jerk_control': False}

    data = create_dataset(v_interval, x_interval, y_interval, psi_interval, planning_kwargs,
                          num_workers=num_workers, data_dir=data_dir, name=dataset_string)

    # Save and Plot Dataset ##################
    used_control = ["v_delta", "acc"]
    labels = intial_state0.state_names + used_control

    print("storing data ...")
    np.savez_compressed(data_dir / (dataset_string + "_complete.npz"), data)
    print(f"...data stored in {data_dir}, create plots and description")

    # Create Text Description of dataset
    with open(data_dir / (dataset_string + ".txt"), "w") as file:
        file.write(f"dynamic_model: {dynamic_model.name}\n"
                   f"vehicle_model: {vehicle_model.name}\n"
                   f"labels: {labels}\n"
                   f"control_inputs: {used_control}\n"
                   f"planning_horizon =  {planning_horizon}\n"
                   f"dt = {dt}\n"
                   f"initial_state: {intial_state0.values}\n"
                   f"v0 in {v_min, v_max}; dv = {dv}\n"
                   f"x_goal range dependent on v0; dx = {dx}\n"
                   f"y_goal in {y_min, y_max}; dy = {dy}\n"
                   f"psi_goal in {psi_min, psi_max}; dpsi = {dpsi}\n"
                   f"dataset dimensions: num_final_points x (num_states + num_u) x num_steps = {data.shape}\n"
                   )
        for v0 in np.unique(data[:, 3, 0]):
            v_data = data[data[:, 3, 0] == v0]
            file.write(f"v0={np.round(v0,2)}: num_points={len(v_data)}, thereby: \n")
            for psi_f in np.unique(np.round(v_data[:, 4, -1], 1)):
                psi_data = v_data[np.round(v_data[:, 4, -1], 1) == psi_f]

                file.write(f"\tpsi={psi_f}: num_points={len(psi_data)}\n"
                           f"\t v_min={min(v_data[:,3,-1])}, v_max={max(v_data[:,3,-1])}\n"
                           f"\t\t x_min={min(v_data[:,0,-1])}, x_max={max(v_data[:,0,-1])}\n")

    print("finished creating dataset")


if __name__ == "__main__":
    main()
