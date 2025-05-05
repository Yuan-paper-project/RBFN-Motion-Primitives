__author__ = "Marc Kaufed"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

import casadi as ca
from vehiclemodels.vehicle_parameters import setup_vehicle_parameters

STATE_LABELS = {
    "point_mass": ["x", "y", "psi", "v", "psi_dot", "a_long"],
    "kinematic_single_track": ["x", "y", "psi", "v", "a_long", "delta"],
    "single_track": ["x", "y", "psi", "v", "psi_dot", "a_long", "delta", "beta"],
}
CONTROL_LABELS = {
    "jerk": ["j_long", "j_lat"],
    "steering_jerk": ["j_long", "v_delta"],
}

VEHICLE_PARAMS = {  # vehicle parameters according to commonroad vehicles
    "FordEscort": setup_vehicle_parameters(1),
    "BMW320i": setup_vehicle_parameters(2),
    "VWVanagon": setup_vehicle_parameters(3),
    "SemiTrailerTruck": setup_vehicle_parameters(4),
}


def point_mass_jerk(x_vec, u_vec, params=None):
    """
    RHS of dynamic model for point mass model for OCP with jerk as control input.
    """
    # states
    # x = x_vec[0] # x position
    # y = x_vec[1] # y position
    psi = x_vec[2]  # yaw angle
    v = x_vec[3]  # velocity
    psi_dot = x_vec[4]  # yaw rate
    a_long = x_vec[5]  # acceleration

    # controls
    j_long = u_vec[0]  # longitudinal jerk
    j_lat = u_vec[1]  # lateral jerk

    # dynamic system
    rhs = ca.horzcat(
        v * ca.cos(psi),  # = x_dot
        v * ca.sin(psi),  # = y_dot
        psi_dot,  # = psi_dot
        a_long,  # = v_dot
        j_lat,  # = psi_dot_dot
        j_long,  # = a_long_dot
    )
    return rhs


def kinematic_single_track_steering_jerk(x_vec, u_vec, params):
    """
    RHS of dynamic model for kinematic single track model for OCP with steering angle and longitudinal jerk as control input.
    """
    # params
    lwb = params[0] + params[1]  # vehicle length
    # states
    # x = x_vec[0] # x position
    # y = x_vec[1] # y position
    psi = x_vec[2]  # yaw angle
    v = x_vec[3]  # velocity
    a_long = x_vec[4]  # acceleration
    delta = x_vec[5]  # steering angle

    # controls
    j_long = u_vec[0]  # longitudinal jerk
    v_delta = u_vec[1]  # steering rate

    # dynamic system
    rhs = ca.horzcat(
        v * ca.cos(psi),  # = x_dot
        v * ca.sin(psi),  # = y_dot
        v / lwb * ca.tan(delta),  # = psi_dot
        a_long,  # = v_dot
        j_long,  # = a_long_dot
        v_delta,  # = delta_dot
    )

    return rhs


def single_track_steering_jerk(x_vec, u_vec, params):
    """
    RHS of dynamic model for single track model for OCP with steering angle and longitudinal jerk as control input.
    """
    # params
    g = 9.81  # gravity
    lf = params[0]  # vehicle length cog - front axle
    lr = params[1]  # vehicle length cog - rear axle
    lwb = lf + lr  # vehicle length
    m = params[2]  # vehicle mass
    h = params[3]  # height of cog
    Iz = params[4]  # moment of inertia

    mu = params[5]  # friction coefficient

    C_Sf = -params[6] / params[5]  # cornering stiffness front axle
    C_Sr = -params[6] / params[5]  # cornering stiffness rear axle

    #
    # states
    # x = x_vec[0] # x position
    # y = x_vec[1] # y position
    psi = x_vec[2]  # yaw angle
    v = x_vec[3]  # velocity
    psi_dot = x_vec[4]  # yaw rate
    a_long = x_vec[5]  # acceleration
    delta = x_vec[6]  # steering angle
    beta = x_vec[7]  # side slip angle

    # controls
    j_long = u_vec[0]  # longitudinal jerk
    v_delta = u_vec[1]  # steering rate

    Cr = C_Sr * (g * lf + a_long * h)
    Cf = C_Sf * (g * lr + a_long * h)

    alpha = 1000
    switch = 1 / (1 + ca.exp(-alpha * (v - 0.1)))  # switch between high and low velocity model
    beta_dot_high_v = mu / (v * lwb) * (Cf * delta - (Cr + Cf) * beta + (lr * Cr - lf * Cf) * psi_dot / v) - psi_dot
    beta_dot_low_v = 1 / (1 + (ca.tan(delta) * lr / lwb) ** 2) * (lr / (lwb * ca.cos(delta) ** 2)) * v_delta
    beta_dot = switch * beta_dot_high_v + (1 - switch) * beta_dot_low_v

    psi_dot_dot_high_v = (
        mu * m / (Iz * lwb) * (lf * Cf * delta + (lr * Cr - lf * Cf) * beta - (lf**2 * Cf + lr**2 * Cr) * psi_dot / v)
    )
    psi_dot_dot_low_v = (
        1
        / lwb
        * (
            a_long * ca.cos(beta) * ca.tan(delta)
            - v * ca.sin(beta) * ca.tan(delta) * beta_dot_low_v
            + (v * ca.cos(beta)) / (ca.cos(delta) ** 2) * v_delta
        )
    )

    psi_dot_dot = switch * psi_dot_dot_high_v + (1 - switch) * psi_dot_dot_low_v

    psi_dot_rhs = switch * psi_dot + (1 - switch) * (v * ca.cos(beta)) / lwb * ca.tan(delta)
    # dynamic system
    rhs = ca.horzcat(
        v * ca.cos(psi + beta),  # = x_dot
        v * ca.sin(psi + beta),  # = y_dot
        psi_dot_rhs,   # = psi_dot
        a_long,  # = v_dot
        psi_dot_dot,  # = psi_dot_dot
        j_long,  # = a_long_dot
        v_delta,  # = delta_dot
        beta_dot,  # = beta_dot
    )

    return rhs


def get_model(vehicle_config):
    rhs_name = vehicle_config["dynamic_model"] + "_" + vehicle_config["control_mode"]
    rhs = globals().get(rhs_name)

    vehicle_params = vehicle_config["vehicle_params"]

    # lf, lr, m , h_cog, mu, Iz, mu, C_Sf, C_Sr
    rhs_params = [
        vehicle_params.a,
        vehicle_params.b,
        vehicle_params.m,
        vehicle_params.h_s,
        vehicle_params.I_z,
        vehicle_params.tire.p_dy1,
        vehicle_params.tire.p_ky1,
    ]

    if not callable(rhs):
        raise ValueError(
            f"Dynamic model {vehicle_config['dynamic_model']} with control mode {vehicle_config['control_mode']} not implemented."
        )
    return {
        "state_labels": STATE_LABELS[vehicle_config["dynamic_model"]],
        "control_labels": CONTROL_LABELS[vehicle_config["control_mode"]],
        "rhs": rhs,
        "rhs_params": rhs_params,
    }
