__author__ = "Marc Kaufed"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import time
from ml_planner.analytic_solution import ivp_optimization_methods as ivp_solvers
from ml_planner.general_utils.vehicle_models import get_model


def solve_ocp(
    num_steps,
    dt,
    num_states,
    num_controls,
    state_constraints,
    control_constraints,
    ic,
    fc,
    rhs_params,
    rhs,
    msg_logger,
    ivp_sol=ivp_solvers.heun,
):
    # timer
    start_time = time.time()
    # optimization instance
    opti = ca.Opti()

    # initialize state variables and set initial and final conditions
    x_vec = opti.variable(num_steps, num_states)
    # initial conditions
    ic_params = np.zeros(num_states)  # needed for initial guess
    for idx, val in ic.items():
        ic_params[idx] = val
        opti.subject_to(x_vec[0, idx] == val)
    # final conditions
    fc_params = ic_params.copy()  # needed for initial guess
    for idx, val in fc.items():
        fc_params[idx] = val
        opti.subject_to(x_vec[-1, idx] == val)

    # initialize control variables and set initial and final conditions
    u_vec = opti.variable(num_steps, num_controls)
    # set final control states to zero
    opti.subject_to(u_vec[-1, :] == 0)

    # state constraints valid for all time steps
    # limit velocity
    v_const = state_constraints["v"]
    v_idx = list(v_const.keys())[0]
    opti.subject_to(x_vec[:, v_idx] >= v_const[v_idx][0])
    opti.subject_to(x_vec[:, v_idx] <= v_const[v_idx][1])

    # acceleration constraints
    a_const = state_constraints["a"]
    [a_idx, a_lat_idx, _] = list(a_const.keys())
    (a_long_min, a_long_max) = a_const[a_idx]
    a_lat_max = a_const[a_lat_idx]
    v_switch = a_const[v_idx]
    # max braking
    opti.subject_to(x_vec[:, a_idx] >= a_long_min) # currently overruled by min total acceleration
    # steering angle
    if "delta" in state_constraints:
        delta_const = state_constraints["delta"]
        delta_idx = list(delta_const.keys())[0]
        opti.subject_to(x_vec[:, delta_idx] >= delta_const[delta_idx][0])
        opti.subject_to(x_vec[:, delta_idx] <= delta_const[delta_idx][1])
    # side slip angle:
    if "beta" in state_constraints:
        beta_const = state_constraints["beta"]
        beta_idx = list(beta_const.keys())[0]
        opti.subject_to(x_vec[:, beta_idx] >= beta_const[beta_idx][0])
        opti.subject_to(x_vec[:, beta_idx] <= beta_const[beta_idx][1])

    # control constraints valid for all time steps
    u = np.zeros(num_controls)
    for key, val in control_constraints.items():
        opti.subject_to(u_vec[:, key] >= val[0])
        opti.subject_to(u_vec[:, key] <= val[1])
        # cost weighting
        u[key] = 1 / val[1]
        # u_1 = 1/val[1]

    # step dependent costs and constraints
    cost = 0
    # cost function
    if "delta" in state_constraints:
        # steering velocity as control sidewards
        for k in range(num_steps - 1):
            cost += (
                u[0] * u_vec[k, 0] ** 2  # j_lat
                + (
                    u[1]
                    * (
                        2
                        * x_vec[k, a_idx]
                        * x_vec[k, v_idx]
                        / (rhs_params[0] + rhs_params[1])
                        * ca.tan(x_vec[k, delta_idx])
                        + x_vec[k, v_idx] ** 2
                        / ((rhs_params[0] + rhs_params[1]) * ca.cos(x_vec[k, delta_idx]) ** 2)
                        * u_vec[k, 1]
                    )  # j_long
                )
                ** 2
            )
    else:
        # lat jerk as control sidewards
        for k in range(num_steps - 1):
            cost += u[0] * u_vec[k, 0] ** 2 + u[1] * u_vec[k, 1] ** 2

    # time step dependent constraints
    for k in range(num_steps - 1):
        # vehicle dynamics
        opti.subject_to(x_vec[k + 1, :] == ivp_sol(rhs, x_vec[k, :], u_vec[k, :], p=rhs_params, h=dt))

        # available long acceleration
        sign = ca.tanh(20 * x_vec[k, a_idx])
        v_ratio = ca.fmin(1, v_switch / (x_vec[k, v_idx] + 1e-6))  # avoid division by zero
        a_long = a_long_max * v_ratio * (1 + sign) / 2 + (-a_long_min) * (1 - sign) / 2
        # min total acceleration based on gg(v) - diagramm
        opti.subject_to(
                (x_vec[k, a_idx] / a_long) ** 2
                + ((x_vec[k, v_idx] ** 2 * ca.tan(x_vec[k, delta_idx]) / (rhs_params[0] + rhs_params[1])) / a_lat_max) ** 2
                <= 1
            )

    # initial guess
    initial_guess = np.linspace(ic_params, fc_params, num_steps)
    opti.set_initial(x_vec, initial_guess)

    opti.minimize(cost)

    ipopt_opts = {
        "print_level": 0,
        "print_timing_statistics": "no",
        "max_iter": 1000,
        "sb": "yes",  # suppress IPOPT banner
        "warm_start_init_point": "yes",
        "warm_start_same_structure": "no",
    }
    opti.solver("ipopt", {"print_time": False, "record_time": True}, ipopt_opts)

    try:
        sol = opti.solve()
        x_sol = sol.value(x_vec)
        u_sol = sol.value(u_vec)
        msg_logger.info(f"from {ic_params} to {fc_params}: solution")
    except RuntimeError:
        msg_logger.info(f"from {ic_params} to {fc_params}: failed")
        x_sol = None
        u_sol = None

        sol = opti.debug
        x_sol = sol.value(x_vec)
        u_sol = sol.value(u_vec)
    stats = opti.stats()
    t_proc = stats["t_proc_total"]
    t_wall = stats["t_wall_total"]
    measured_time = time.time() - start_time

    return x_sol, u_sol, [measured_time, t_proc, t_wall]


def draw_trajectories(x_vec, u_vec, state_labels, control_labels):
    nfig = x_vec.shape[1] + 1
    nrow = int(np.ceil(nfig / 2))
    ncol = 2
    fig, axs = plt.subplots(nrow, ncol, figsize=(10, 5))

    axes = axs.flatten()
    axes[0].plot(x_vec[:, 0], x_vec[:, 1])
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_ylim(min(x_vec[:, 1]) - 1, max(x_vec[:, 1]) + 1)

    for i in range(2, x_vec.shape[1]):
        axes[i - 1].plot(x_vec[:, i])
        axes[i - 1].set_xlabel("t")
        axes[i - 1].set_ylabel(state_labels[i])
        axes[i - 1].set_ylim(min(x_vec[:, i]) - 0.2, max(x_vec[:, i]) + 0.2)

    for i in range(u_vec.shape[1]):
        axes[-2 + i].plot(u_vec[:, i])
        axes[-2 + i].set_xlabel("t")
        axes[-2 + i].set_ylabel(control_labels[i])
        axes[-2 + i].set_ylim(min(u_vec[:, i]) - 0.5, max(u_vec[:, i]) + 0.5)
    plt.show()
    pass


def main(planning_config, vehicle_config, msg_logger, ivp_solver="heun", draw=False):
    start_time = time.time()
    # vehicle and dynamic model configuration
    model_config = get_model(vehicle_config)

    state_labels = model_config["state_labels"]
    msg_logger.debug(f"state labels: {state_labels}")
    control_labels = model_config["control_labels"]
    # rhs of dynamic model
    rhs = model_config["rhs"]
    rhs_params = model_config["rhs_params"]
    # vehicle parameters
    vehicle_params = vehicle_config["vehicle_params"]

    # planning configuration
    T = planning_config["T"]  # planning horizon
    dt = planning_config["dt"]  # step size
    num_steps = int(T / dt) + 1  # number of steps

    # solver method for IVP
    ivp_sol = getattr(ivp_solvers, ivp_solver)

    # boundary conditions:
    initial_conditions = {
        state_labels.index("x"): planning_config["x0"],
        state_labels.index("y"): planning_config["y0"],
        state_labels.index("psi"): planning_config["psi0"],
        state_labels.index("v"): planning_config["v0"],  # to be looped
    }

    final_conditions = {
        state_labels.index("x"): planning_config["xf"],  # to be looped
        state_labels.index("y"): planning_config["yf"],  # to be looped
        state_labels.index("psi"): planning_config["psif"],  # to be looped
        state_labels.index("a_long"): planning_config["a_longf"],
    }

    state_constraints = {
        "v": {state_labels.index("v"): (0, vehicle_params.longitudinal.v_max)},
        "a": {
            state_labels.index("a_long"): (
                -vehicle_params.longitudinal.a_max,
                vehicle_params.longitudinal.a_max,
            ),  # a_long
            42: vehicle_params.a_lat,  # a_lat
            state_labels.index("v"): vehicle_params.longitudinal.v_switch,
        },
    }

    if "delta" in state_labels:
        # if not point mass model
        initial_conditions[state_labels.index("delta")] = planning_config["delta0"]  # to be looped
        final_conditions[state_labels.index("delta")] = planning_config["deltaf"]
        state_constraints["delta"] = {
            state_labels.index("delta"): (vehicle_params.steering.min, vehicle_params.steering.max),
        }
    if "beta" in state_labels:
        final_conditions[state_labels.index("beta")] = planning_config["betaf"]
        state_constraints["beta"] = {
            state_labels.index("beta"): (-np.pi / 8, np.pi / 8),
        }

    if "v_delta" in control_labels:
        control_constraints = {
            control_labels.index("j_long"): (-10, 10),
            control_labels.index("v_delta"): (vehicle_params.steering.v_min, vehicle_params.steering.v_max),
        }
    else:
        control_constraints = {
            control_labels.index("j_long"): (-10, 10),
            control_labels.index("j_lat"): (-10, 10),
        }

    x_vec, u_vec, solver_time = solve_ocp(
        num_steps=num_steps,
        dt=dt,
        num_states=len(state_labels),
        num_controls=len(control_labels),
        state_constraints=state_constraints,
        control_constraints=control_constraints,
        ic=initial_conditions,
        fc=final_conditions,
        rhs=rhs,
        rhs_params=rhs_params,
        msg_logger=msg_logger,
        ivp_sol=ivp_sol,
    )
    measured_time = [time.time() - start_time]
    if x_vec is not None and u_vec is not None and draw is True:
        draw_trajectories(x_vec, u_vec, state_labels, control_labels)
    return x_vec, u_vec, measured_time + solver_time


if __name__ == "__main__":

    from pathlib import Path
    from ml_planner.general_utils.logging import logger_initialization
    from ml_planner.general_utils.vehicle_models import VEHICLE_PARAMS

    ###############################
    # PATH AND DEBUG CONFIGURATION
    CWD = Path.cwd()
    LOG_PATH = CWD / "logs"

    LOGGING_LEVEL = "info"

    # logging
    msg_logger = logger_initialization(
        LOG_PATH, "Dataset_Creation", loglevel=LOGGING_LEVEL.upper(), loglevel_msg="INFO"
    )

    planning_config = {
        "T": 3,
        "dt": 0.1,
        "x0": 0,
        "y0": 0,
        "psi0": 0,
        "v0": 2,  # to be looped
        "delta0": 0.00,  # to be looped
        "xf": 25,  # to be looped
        "yf": 0,  # to be looped
        "psif": 0,  # to be looped
        "a_longf": 0,
        "deltaf": 0,
        "betaf": 0,
    }
    vehicle_config = {
        "vehicle_type": "BMW320i",
        "dynamic_model": "kinematic_single_track",  # "point_mass", "kinematic_single_track", "single_track"
        "control_mode": "steering_jerk",  # "jerk",  "steering_jerk"
    }

    vehicle_config["vehicle_params"] = VEHICLE_PARAMS[vehicle_config["vehicle_type"]]
    vehicle_config["vehicle_params"].a_lat = 0.5 * 9.81

    _, _, times = main(planning_config, vehicle_config, msg_logger, ivp_solver="heun", draw=True)
    print(f"Total time: {times[0]} s, solver time: {times[1]} s, t_proc: {times[2]} s, t_wall: {times[3]} s")

