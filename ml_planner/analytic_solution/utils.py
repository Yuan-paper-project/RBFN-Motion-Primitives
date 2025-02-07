__author__ = "Marc Kaufeld"
__copyright__ = "TUM Professorship Autonomous Vehicle Systems"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

import numpy as np
from omegaconf import OmegaConf
from dataclasses import astuple

from vehiclemodels.init_ks import init_ks
from vehiclemodels.init_kst import init_kst
from vehiclemodels.init_mb import init_mb
from vehiclemodels.init_st import init_st
from vehiclemodels.init_std import init_std


from vehiclemodels.vehicle_dynamics_st import vehicle_dynamics_st
from vehiclemodels.vehicle_parameters import VehicleParameters

from ml_planner.analytic_solution.vehicle_dynamics_ks import vehicle_dynamics_ks, der_vehicle_dynamics_ks
from ml_planner.analytic_solution.vehicle_dynamics_ks_jerkcontrol import (vehicle_dynamics_ks_jerk,
                                                                          der_vehicle_dynamics_ks_jerk)
from ml_planner.helper_functions.data_types import StateVector, DynamicModel, VehicleModel


def get_model_state_vector(state_vector: StateVector, dynamic_model: DynamicModel, vehicle_params: VehicleParameters):
    """Returns the state vector for the given dynamic model"""
    if dynamic_model == DynamicModel.KS:
        # Kinematic Single Track Model
        x0_vec = init_ks(astuple(state_vector))
        state_vector.psi_d = None
        state_vector.beta = None
        state_vector.alpha = None

    elif dynamic_model == DynamicModel.KST:
        # Kinematic Single Track Model with trailer
        assert state_vector.alpha, "state_vector requires hatch angle alpha for KST model"
        assert dynamic_model == VehicleModel.SemiTrailerTruck, \
            "KST model is only available for SemiTrailerTruck"
        state_vector.psi_d = None
        state_vector.beta = None
        x0_vec = init_kst(astuple(state_vector), state_vector.alpha)

    elif dynamic_model == DynamicModel.MB:
        # Multi Body Model
        assert state_vector.psi_d is not None and state_vector.beta is not None, \
            "state_vector requires psi_d and beta for MB model"
        state_vector.alpha = None
        x0_vec = init_mb(astuple(state_vector)[:7], vehicle_params)

    elif dynamic_model == DynamicModel.ST:
        # Single Track Model
        assert state_vector.psi_d is not None and state_vector.beta is not None, \
            "state_vector requires psi_d and beta for ST model"
        state_vector.alpha = None
        x0_vec = init_st(astuple(state_vector)[:7])

    elif dynamic_model == DynamicModel.STD:
        # Single Track Model with Drifting
        assert state_vector.psi_d is not None and state_vector.beta is not None, \
            "state_vector requires psi_d and beta for MB model"
        state_vector.alpha = None
        x0_vec = init_std(list(astuple(state_vector)[:7]), vehicle_params)
        state_vector.v_ang_front_wheel = x0_vec[7]
        state_vector.v_ang_rear_wheel = x0_vec[8]
    return np.array(x0_vec), state_vector


def func_KS(t, x, u, p):
    """Vehicle dynamics function for Kinematic Single Track Model"""
    f = vehicle_dynamics_ks(x, u, p)
    return f


def der_func_KS(t, x, u, p):
    """Jacobian of kinematic single-track vehicle dynamics"""
    df = der_vehicle_dynamics_ks(x, u, p)
    return df


def func_KS_jerk(t, x, u, p):
    """Vehicle dynamics function for Kinematic Single Track Model with jerk control"""
    f = vehicle_dynamics_ks_jerk(x, u, p)
    return f


def der_func_KS_jerk(t, x, u, p):
    """Jacobian of kinematic single-track vehicle dynamics with jerk control"""
    df = der_vehicle_dynamics_ks_jerk(x, u, p)
    return df


def func_ST(t, x, u, p):
    """Vehicle dynamics function for Single Track Model"""
    p = OmegaConf.create(p)
    f = vehicle_dynamics_st(x, u, p)
    return np.array(f)


def get_vehicle_dynamics_system(dynamic_model: DynamicModel, vehicle_params: VehicleParameters, jerk_control=False):
    """Returns the vehicle dynamics function and its jacobian for the given dynamic model"""
    if dynamic_model == DynamicModel.KS:
        p = [[vehicle_params.a, vehicle_params.b],
             [vehicle_params.steering.min, vehicle_params.steering.max,
              vehicle_params.steering.v_min, vehicle_params.steering.v_max],
             [vehicle_params.longitudinal.a_max, vehicle_params.longitudinal.v_switch,
              vehicle_params.longitudinal.v_min, vehicle_params.longitudinal.v_max]]
        if jerk_control:
            return func_KS_jerk, der_func_KS_jerk, p
        else:
            return func_KS, der_func_KS, p
    elif dynamic_model == DynamicModel.KST:
        raise NotImplementedError("Params for KST model not implemented")
        # return func_KST
    elif dynamic_model == DynamicModel.ST:
        raise NotImplementedError("Params for ST model not implemented")
        # return func_ST
    elif dynamic_model == DynamicModel.MB:
        raise NotImplementedError("Params for MB model not implemented")
        # return func_MB
    elif dynamic_model == DynamicModel.STD:
        raise NotImplementedError("Params for STD model not implemented")
        # return func_STD


def vehicle_output(t, x, u, p):
    """Vehicle output function"""
    return x


def get_vehicle_output_func():
    """Returns the vehicle output function"""
    f = vehicle_output
    return f
