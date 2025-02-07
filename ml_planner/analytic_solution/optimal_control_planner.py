__author__ = "Marc Kaufeld"
__copyright__ = "TUM Professorship Autonomous Vehicle Systems"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

import logging
import numpy as np
from typing import List, Union
import matplotlib.pyplot as plt
import control as ct

from vehiclemodels.vehicle_parameters import setup_vehicle_parameters
from vehiclemodels.utils.steering_constraints import steering_constraints
from vehiclemodels.utils.acceleration_constraints import acceleration_constraints

from ml_planner.helper_functions.data_types import StateVector, ControlVector, DynamicModel, VehicleModel

from ml_planner.analytic_solution.utils import (get_model_state_vector,
                                                get_vehicle_dynamics_system,
                                                get_vehicle_output_func)
from ml_planner.analytic_solution.optimal_control_problem import (state_range_constraint,
                                                                  input_range_constraint,
                                                                  quadratic_cost, solve_ocp)

logger = logging.getLogger("DiffSampling")


class OCPlanner:
    def __init__(self,
                 dt: float = 0.1,
                 dynamic_model: DynamicModel = DynamicModel.KS,
                 vehicle_model: VehicleModel = VehicleModel.BMW320i,
                 jerk_control: bool = False,
                 plot_dir=None):
        self.plot_dir = plot_dir
        self.jerk_control = jerk_control
        self._validate_input(dt, dynamic_model, vehicle_model)
        self.dt = dt
        # vehicle model & parameters
        self.vehicle_model = vehicle_model
        self.dynamic_model = dynamic_model
        self.vehicle_params = setup_vehicle_parameters(self.vehicle_model.value)
        # vehicle dynamics system
        (self._dyn_mod_sys,
         self._der_dyn_mod_sys,
         self.sys_param_list) = get_vehicle_dynamics_system(self.dynamic_model,
                                                            self.vehicle_params,
                                                            jerk_control=jerk_control)
        # output function of the vehicle dynamics system
        self._outputfcn = get_vehicle_output_func()
        # create ct dynamics system
        self.iosys = ct.NonlinearIOSystem(updfcn=self._dyn_mod_sys,
                                          outfcn=self._outputfcn,
                                          name='dynamic_model',
                                          params=self.sys_param_list,
                                          )
        self.iosys.d_rhs = self._der_dyn_mod_sys
        # initialize variables
        self.x0 = self.xF = None
        self.x_mat = self.u = None
        self.init_state = self.goal_state = self.control_vector = None
        logger.debug(f"Initialized vehicle model: {self.dynamic_model.name}")

    def _validate_input(self,  dt, dynamic_model, vehicle_model):
        assert isinstance(dt, float) and dt > 0, "dt must be a float greater than 0"
        assert isinstance(dynamic_model, DynamicModel), "dynamic_model must be an instance of DynamicModel"
        assert isinstance(vehicle_model, VehicleModel), \
            "vehicle_model must be an instance of VehicleModel"

    def _set_traj_constraints(self):
        """trajectory constraints"""
        if self.jerk_control:
            # min jerl ocp
            # state constraints x, y, delta, v, psi, v_delta, acc, :
            lb = [-np.inf, -np.inf, self.vehicle_params.steering.min, 0, -np.pi,
                  self.vehicle_params.steering.v_min, -self.vehicle_params.longitudinal.a_max]
            lb += [-np.inf] * (len(self.x0)-len(lb))  # append pseudo constraints for unconstrained states if they exist
            ub = [np.inf, np.inf, self.vehicle_params.steering.max, self.vehicle_params.longitudinal.v_max, np.pi,
                  self.vehicle_params.steering.v_max, self.vehicle_params.longitudinal.a_max]
            ub += [np.inf] * (len(self.x0)-len(ub))
            state_const = state_range_constraint(self.iosys, lb=lb, ub=ub)

            # input constraints steering_acc, jerk:
            lb = [-np.inf, -self.vehicle_params.longitudinal.j_max]
            ub = [np.inf, self.vehicle_params.longitudinal.j_max]
            input_const = input_range_constraint(self.iosys, lb=lb, ub=ub)
        else:
            # min acc ocp
            # state constraints x, y, delta, v, psi :
            lb = [-np.inf, -np.inf, self.vehicle_params.steering.min, 0, -np.pi]
            lb += [-np.inf] * (len(self.x0)-len(lb))  # append pseudo constraints for unconstrained states if they exist
            ub = [np.inf, np.inf, self.vehicle_params.steering.max, self.vehicle_params.longitudinal.v_max, np.pi]
            ub += [np.inf] * (len(self.x0)-len(ub))
            state_const = state_range_constraint(self.iosys, lb=lb, ub=ub)

            # input constraints v_delta, acc:
            lb = [self.vehicle_params.steering.v_min, -self.vehicle_params.longitudinal.a_max]
            ub = [self.vehicle_params.steering.v_max, self.vehicle_params.longitudinal.a_max]
            input_const = input_range_constraint(self.iosys, lb=lb, ub=ub)
        return [state_const, input_const]

    def _set_terminal_constraints(self, u0: np.ndarray = np.array([0, 0]), relax_state_const: List[int] = []):
        """terminal constraints"""
        ub = self.xF.copy()
        lb = self.xF.copy()
        for i in relax_state_const:
            # relax constraints for states that are not considered in the cost
            ub[i] += 100
            lb[i] -= 100
        term_const = [state_range_constraint(self.iosys, lb=lb, ub=ub),
                      input_range_constraint(self.iosys, lb=u0, ub=u0)]
        return term_const

    def _set_traj_cost(self, u0: np.ndarray = np.array([0, 0]), wu: list = [1000, 10],
                       wx: float = 0):
        """trajectory cost"""
        Q = np.eye(len(self.x0))*wx
        # penalize steering more than acceleration
        R = np.eye(len(u0))*wu
        traj_cost, d_traj_cost = quadratic_cost(sys=self.iosys,
                                                Q=Q,
                                                R=R,
                                                x0=self.xF,
                                                u0=u0)
        return traj_cost, d_traj_cost

    def _set_terminal_cost(self, u0: np.ndarray = np.array([0, 0]), wu: float = 5,
                           wx: float = 5):
        """terminal cost"""
        Q = np.eye(len(self.x0))*wx
        R = np.eye(len(u0))*wu
        term_cost, d_term_cost = quadratic_cost(sys=self.iosys,
                                                Q=Q,
                                                R=R,
                                                x0=self.xF,
                                                u0=u0)
        return term_cost, d_term_cost

    def _initialize_guess(self, t_eval: np.ndarray, time_horizon: float, u0: np.ndarray = np.array([0, 0])):
        """initialize guess for trajectory optimization"""
        initial_curve = np.array([self.x0 + (self.xF - self.x0) * time/time_horizon
                                  for time in t_eval]).transpose()
        initial_contol = np.outer(u0, np.ones_like(t_eval))
        return (initial_curve, initial_contol)

    def plan_to_state(self, initial_state: StateVector, goal_state: StateVector, time_horizon: float = 3):
        """Plan trajectory to reach goal state"""
        # initial state
        self.x0, self.init_state = get_model_state_vector(state_vector=initial_state,
                                                          dynamic_model=self.dynamic_model,
                                                          vehicle_params=self.vehicle_params)
        # final state
        self.xF, self.goal_state = get_model_state_vector(state_vector=goal_state,
                                                          dynamic_model=self.dynamic_model,
                                                          vehicle_params=self.vehicle_params)
        if not self.jerk_control:
            # min acc ocp
            self.control_vector = ControlVector(v_delta=[0], acc=[0])
            self.state_names = self.init_state.state_names
            self.control_names = self.control_vector.control_names

        else:
            # min jerk ocp
            self.x0 = np.hstack((self.x0, [0, 0]))
            self.xF = np.hstack((self.xF, [0, 0]))
            self.state_names = self.init_state.state_names + ["v_delta", "acc"]
            self.control_names = ["a_delta", "jerk"]

        # To avoid singularities
        if self.x0[3] == 0:
            self.x0[3] = 0.001

        self.iosys.set_states(self.init_state.state_names)
        self.iosys.set_inputs(self.control_vector.control_names)
        self.iosys.set_outputs(self.init_state.state_names)

        num_steps = int(time_horizon/self.dt)+1
        t_eval = np.linspace(0, time_horizon, num_steps, endpoint=True)

        # constraints:
        traj_const = self._set_traj_constraints()
        # velocity at final state does not have to be constrained
        term_const = self._set_terminal_constraints(relax_state_const=[3])

        # control input costs -> minimize control inputs
        traj_cost, d_traj_cost = self._set_traj_cost()
        term_cost, d_term_cost = self._set_terminal_cost()

        # initial guess
        initial_guess = self._initialize_guess(t_eval, time_horizon)

        basis_func = None

        result = solve_ocp(sys=self.iosys,
                           timepts=t_eval,
                           X0=self.x0,
                           cost=traj_cost,
                           cost_derivative=d_traj_cost,
                           # terminal_cost= term_cost,
                           # terminal_cost_derivative= d_term_cost,
                           trajectory_constraints=traj_const,
                           terminal_constraints=term_const,
                           use_constraint_derivative=True,
                           initial_guess=initial_guess,
                           basis=basis_func,
                           print_summary=False,
                           log=False,
                           minimize_options={'disp': False,
                                             'maxiter': 500,
                                             'ftol': 1e-6,
                                             },
                           )
        valid = False
        if result.success:
            self.u = result.inputs
            self.x_mat = result.states
            # check feasibility of control inputs
            valid = True
            if self.jerk_control:
                v_delta_const = [steering_constraints(steering_angle=self.x_mat[2][i],
                                                      steering_velocity=self.x_mat[5][i],
                                                      p=self.vehicle_params.steering)
                                 for i in range(len(self.x_mat[5]))]
                acc_const = [acceleration_constraints(velocity=self.x_mat[3][i],
                                                      acceleration=self.x_mat[6][i],
                                                      p=self.vehicle_params.longitudinal)
                             for i in range(len(self.x_mat[6]))]
                v_delta_constr = np.round(self.x_mat[5] - v_delta_const, 4)
                acc_constr = np.round(self.x_mat[6] - acc_const, 4)
            else:
                v_delta_const = [steering_constraints(steering_angle=self.x_mat[2][i],
                                                      steering_velocity=self.u[0][i],
                                                      p=self.vehicle_params.steering)
                                 for i in range(len(self.u[0]))]
                acc_const = [acceleration_constraints(velocity=self.x_mat[3][i],
                                                      acceleration=self.u[1][i],
                                                      p=self.vehicle_params.longitudinal)
                             for i in range(len(self.u[1]))]
                v_delta_constr = np.round(self.u[0] - v_delta_const, 4)
                acc_constr = np.round(self.u[1] - acc_const, 4)
            if any(v_delta_constr) > 0.001:
                logger.warning(f"""Steering constraints violated: by {self.u[0]-v_delta_const} \n
                               allowed: {v_delta_const} \n
                               goal state: {self.goal_state.values}""")
                # self.draw_trajectory(save=True, show=False,
                #                      title=f"Steering constraints violated, x0: {self.x0}, xF: {self.xF}")
                valid = False

            if any(acc_constr) > 0.001:
                logger.warning(f"""Acceleration constraints violated: by {self.u[1]-acc_const}, \n
                               allowed: {acc_const} \n
                               goal state: {self.goal_state.values}""")
                # self.draw_trajectory(save=True, show=False,
                #                      title=f"Acc constraints violated, xF: {self.xF}")
                valid = False
        if valid:

            t = result.time
            # to ensure that trajectorie is feasible, plan it with the control inputs
            self.plan_by_controls(initial_state, self.u, t)
        else:
            self.u = None
            self.x_mat = None
        return

    def plan_by_controls(self, initial_state: StateVector,
                         controls: Union[ControlVector, np.ndarray],
                         T: Union[List[float], None] = None):
        """Plan trajectory by applying given control inputs"""
        def event_stop(t, y):
            # stop solver if vehicle reaches a still stand
            # condition is triggered if event returns 0
            cond = y[3] >= 0
            return cond
        event_stop.terminal = True

        if isinstance(controls, ControlVector):
            assert controls.dt == self.dt, "dt of controls must match dt of planner"
            self.u = controls.u.transpose()
            t_eval = np.linspace(0, controls.time_horizon, controls.num_steps, endpoint=True)
        elif isinstance(controls, np.ndarray):
            self.u = controls
            num_steps = int((T[-1]-T[0])/self.dt)+1
            t_eval = np.linspace(T[0], T[-1], num_steps, endpoint=True)
        else:
            raise ValueError("controls must be an instance of ControlVector or a numpy array")

        # initial state
        self.x0, self.init_state = get_model_state_vector(state_vector=initial_state,
                                                          dynamic_model=self.dynamic_model,
                                                          vehicle_params=self.vehicle_params)
        # To avoid event trigger if starting with v0 = 0
        if self.x0[3] == 0:
            self.x0[3] = 0.001

        # io system
        if not self.iosys.nstates:
            self.iosys.set_states(self.init_state.state_names)
            self.iosys.set_inputs(controls.control_names)
            self.iosys.set_outputs(self.init_state.state_names)
        result = ct.input_output_response(sys=self.iosys,
                                          T=t_eval if T is None else T,  # timesteps at which input is defined
                                          t_eval=t_eval,  # timesteps at which output is evaluated
                                          U=self.u,
                                          X0=self.x0,
                                          solve_ivp_method='LSODA',
                                          solve_ivp_kwargs={"max_step": self.dt/10,
                                                            "events": event_stop,
                                                            "rtol": 1e-6,
                                                            "atol": 1e-6,
                                                            }
                                          )

        if result.success:
            if (len(t_eval) - result.outputs.shape[1]) > 0:
                # termination event occurred if shapes don't match , i.e. vehicle stopped
                final_state = np.tile(result.outputs[:, -1], (len(t_eval) - result.outputs.shape[1], 1)).T
                # velocity is 0 after stopping
                final_state[3, :] = 0
                if len(final_state) > 4:
                    # yaw rate, side slip, and wheel speeds are 0 after stopping when included in the model
                    final_state[5:, :] = 0
                self.x_mat = np.round(np.hstack((result.outputs, final_state)), 4)
                final_input = np.tile(result.inputs[:, -1], (len(t_eval) - result.outputs.shape[1], 1)).T
                self.u = np.round(np.hstack((result.inputs, final_input)), 4)
            else:
                self.x_mat = np.round(result.outputs, 4)
                self.u = np.round(result.inputs, 4)
            self.xF = self.x_mat[:, -1]

            logger.debug("Trajectory planned successfully")
        else:
            # integration step failed
            raise Exception(f"Trajectory planning failed with status '{result.status}' and message '{result.message}'")
        return

    def min_acc_zero_delta(self, initial_state: StateVector, time_horizon: float = 3):
        """Plan trajectory with minimum acceleration and zero steering angle velocity"""
        control_input = ControlVector(v_delta=0,
                                      acc=-self.vehicle_params.longitudinal.a_max,
                                      dt=self.dt,
                                      time_horizon=time_horizon
                                      )
        self.plan_by_controls(initial_state, control_input)

    def max_acc_zero_delta(self, initial_state: StateVector, time_horizon: float = 3):
        """Plan trajectory with maximum acceleration and zero steering angle velocity"""
        control_input = ControlVector(v_delta=0,
                                      acc=self.vehicle_params.longitudinal.a_max,
                                      dt=self.dt,
                                      time_horizon=time_horizon
                                      )
        self.plan_by_controls(initial_state, control_input)

    def zero_acc_max_delta(self, initial_state: StateVector, time_horizon: float = 3):
        """Plan trajectory with zero acceleration and maximum steering angle velocity"""
        control_input = ControlVector(v_delta=self.vehicle_params.steering.v_max,
                                      acc=0,
                                      dt=self.dt,
                                      time_horizon=time_horizon
                                      )
        self.plan_by_controls(initial_state, control_input)

    def zero_acc_min_delta(self, initial_state: StateVector, time_horizon: float = 3):
        """Plan trajectory with zero acceleration and minimum steering angle velocity"""
        control_input = ControlVector(v_delta=self.vehicle_params.steering.v_min,
                                      acc=0,
                                      dt=self.dt,
                                      time_horizon=time_horizon
                                      )
        self.plan_by_controls(initial_state, control_input)

    def min_acc_max_delta(self, initial_state: StateVector, time_horizon: float = 3):
        """Plan trajectory with minimum acceleration and maximum steering angle velocity"""
        control_input = ControlVector(v_delta=self.vehicle_params.steering.v_max,
                                      acc=-self.vehicle_params.longitudinal.a_max,
                                      dt=self.dt,
                                      time_horizon=time_horizon
                                      )
        self.plan_by_controls(initial_state, control_input)

    def draw_trajectory(self, save=False, show=True, block=True, title=None):
        """Draw trajectory of planned path"""
        time = np.linspace(0, self.dt * (len(self.x_mat[0])-1), len(self.x_mat[0]), endpoint=True)
        line_style = 'b-'
        ncol = 2
        nrow = min(int(np.ceil((len(self.x_mat)-1)/ncol)) + 1, 4)
        fig, axs = plt.subplots(nrow, ncol)
        fig.set_size_inches(20, 20, forward=True)
        fig.set_dpi(100)
        title = title if title is not None else f"Trajectory of {self.dynamic_model.name} model"
        fig.suptitle(title, fontsize=16)

        # plot state vector *************
        # Plot x and y position
        axs[0, 0].plot(self.x0[0], self.x0[1], 'ro')
        axs[0, 0].plot(self.x_mat[0], self.x_mat[1], line_style)
        axs[0, 0].set_title('X-Y Position')
        axs[0, 0].set_xlabel('X Position')
        axs[0, 0].set_ylabel('Y Position')

        # Plot yaw angle psi over time
        axs[0, 1].plot(time, self.x_mat[4], line_style)
        axs[0, 1].set_title('Yaw Angle over Time')
        axs[0, 1].set_xlabel('Time [s]')
        axs[0, 1].set_ylabel('Yaw Angle [rad]')

        # Plot yaw rate psi_d over time if present
        if self.init_state.psi_d is not None:
            axs[1, 1].plot(time, self.x_mat[5], line_style)
            axs[1, 1].set_title('Yaw Rate over Time')
            axs[1, 1].set_xlabel('Time [s]')
            axs[1, 1].set_ylabel('Yaw Rate [rad/s]')

        # Plot sideslip angle beta over time if present
        if self.init_state.beta is not None:
            axs[1, 0].plot(time, self.x_mat[6], line_style)
            axs[1, 0].set_title('Sideslip Angle over Time')
            axs[1, 0].set_xlabel('Time [s]')
            axs[1, 0].set_ylabel('Sideslip Angle [rad]')

        i = 1 if self.jerk_control else 0
        # Plot steering angle delta over time
        axs[-2-i, 1].plot(time, self.x_mat[2], line_style)
        axs[-2-i, 1].set_title('Steering Angle over Time')
        axs[-2-i, 1].set_xlabel('Time [s]')
        axs[-2-i, 1].set_ylabel('Steering Angle [rad]')

        # Plot velocity over time
        axs[-2-i, 0].plot(time, self.x_mat[3], line_style)
        axs[-2-i, 0].set_title('Velocity over Time')
        axs[-2-i, 0].set_xlabel('Time [s]')
        axs[-2-i, 0].set_ylabel('Velocity [m/s]')

        # Plot hitch angle alpha over time if present
        if self.init_state.alpha is not None:
            raise NotImplementedError
            # fig, ax = plt.subplots()
            # ax.plot(time, self.x_mat[7], line_style)
            # ax.set_title('Hitch Angle over Time')
            # ax.set_xlabel('Time [s]')
            # ax.set_ylabel('Hitch Angle [rad]')

        # plot control values *************
        # Plot acceleration over time
        if not self.jerk_control:
            acc_input = self.u[1]
        else:
            acc_input = self.x_mat[6]
        acc_input = np.interp(time, np.linspace(0, time[-1], len(acc_input), endpoint=True), acc_input)
        acc_actual = [acceleration_constraints(velocity=self.x_mat[3][i],
                                               acceleration=acc_input[i],
                                               p=self.vehicle_params.longitudinal)
                      for i in range(len(acc_input))]
        axs[-1-i, 0].plot(time, acc_input, 'r-', label='Input')
        axs[-1-i, 0].plot(time, acc_actual, line_style, label='Actual with vehicle constraints')
        axs[-1-i, 0].set_title('Acceleration over Time')
        axs[-1-i, 0].set_xlabel('Time [s]')
        axs[-1-i, 0].set_ylabel('Acceleration [m/s²]')

        # Plot steering angle velocity over time
        if not self.jerk_control:
            v_delta_input = self.u[0]
        else:
            v_delta_input = self.x_mat[5]
        # v_delta_input = self.u[0]
        v_delta_input = np.interp(time, np.linspace(0, time[-1], len(v_delta_input), endpoint=True), v_delta_input)
        v_delta_actual = [steering_constraints(steering_angle=self.x_mat[2][i],
                                               steering_velocity=v_delta_input[i],
                                               p=self.vehicle_params.steering)
                          for i in range(len(v_delta_input))]
        axs[-1-i, 1].plot(time, v_delta_input, 'r-', label='Input')
        axs[-1-i, 1].plot(time, v_delta_actual, line_style, label='Actual with vehicle constraints')
        axs[-1-i, 1].set_title('steering angle velocity over Time')
        axs[-1-i, 1].set_xlabel('Time [s]')
        axs[-1-i, 1].set_ylabel('steering angle velocity [rad/s]')
        axs[-1-i, 1].legend()

        if self.jerk_control:
            # plot jerk
            axs[-1, 0].plot(time, self.u[1], line_style, label='Input')
            axs[-1, 0].set_title('Jerk over Time')
            axs[-1, 0].set_xlabel('Time [s]')
            axs[-1, 0].set_ylabel('Jerk [m/s³]')

            # plot steering acceleration over time
            axs[-1, 1].plot(time, self.u[0], line_style, label='Input')
            axs[-1, 1].set_title('steering angle acceleration over Time')
            axs[-1, 1].set_xlabel('Time [s]')
            axs[-1, 1].set_ylabel('steering angle velocity [rad/s²]')

        if save:
            dic = self.plot_dir / (title + ".svg")
            plt.savefig(dic, format="svg")
        if show:
            plt.show(block=block)
        plt.close()

        return


# Test Planner *****************
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG, filename="dev.log",
        filemode='w', force=True)

    dt = 0.1  # time step size

    dynamic_model = DynamicModel.KS
    vehicle_model = VehicleModel.BMW320i

    x0 = 0.  # x position
    y0 = 0.  # y position
    delta0 = 0  # steering angle
    vel0 = 10.  # velocity
    psi0 = 0  # yaw angle
    psi_d0 = 0.  # yaw rate
    beta0 = 0.  # sideslip angle at vehicle center
    alpha0 = 0.  # hitch angle if ks with trailer is used

    initial_state = StateVector(x=x0,
                                y=y0,
                                delta=delta0,
                                v=vel0,
                                psi=psi0,
                                psi_d=psi_d0,
                                beta=beta0,
                                alpha=alpha0
                                )

    planner = OCPlanner(dt=dt, dynamic_model=dynamic_model,
                        vehicle_model=vehicle_model)

    control_input = ControlVector(v_delta=[1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0],
                                  acc=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 4, 3, 2, 1, 0, 0, 0, -1,
                                       -1, 0, 0],
                                  dt=dt,
                                  time_horizon=10)

    goal_state = StateVector(x=x0+52,
                             y=y0-7.5,
                             delta=delta0,
                             v=vel0,
                             psi=psi0,
                             psi_d=psi_d0,
                             beta=beta0,
                             alpha=alpha0
                             )
    # planner.min_acc_zero_delta(initial_state, time_horizon=3)
    # planner.max_acc_zero_delta(initial_state, time_horizon=3)
    # planner.zero_acc_max_delta(initial_state, time_horizon=3)
    # planner.min_acc_max_delta(initial_state, time_horizon=3)
    # planner.plan_by_controls(initial_state, control_input)
    # planner.draw_trajectory()

    planner.plan_to_state(initial_state, goal_state, time_horizon=3)
    planner.draw_trajectory(save=True)

    # import cProfile
    # import pstats
    # import pandas as pd
    # # import gprof2dot
    # # import snakeviz

    # profiler = cProfile.Profile()
    # profiler.enable()
    # planner.plan_to_state(initial_state, goal_state, time_horizon=3)
    # profiler.disable()
    # profiler.dump_stats("train_profile.prof")

    # planner.draw_trajectory()

    # # Analyzing the profile data
    # p = pstats.Stats("train_profile.prof")
    # p.strip_dirs()
    # p.sort_stats('cumulative')  # Sort by cumulative time
    # p.print_stats(50) # prints first 500 entries
