__author__ = "Marc Kaufeld"
__copyright__ = "TUM Professorship Autonomous Vehicle Systems"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

import numpy as np
from enum import Enum
from typing import Optional, Union, List
from dataclasses import dataclass, field
import logging

logger = logging.getLogger("DiffSampling")


class DynamicModel(Enum):
    """
    Enum class representing different types of (Commonroad) dynamic models.

    Attributes:
        KS (int): Kinematic Single Track Model.
        KST (int): Kinematic Single Track Model with Trailer.
        MB (int): Multi-body Model.
        ST (int): Linear Single Track Model.
        STD (int): Drift Single Track Model.
    """
    KS = 1  # Kinematic Single Track Model
    KST = 2  # Kinematic Single Track Model with Trailer
    MB = 3  # Multi-body Model
    ST = 4  # Linear Single Track Model
    STD = 5  # Drift Single Track Model


class VehicleModel(Enum):
    """
    Enum class representing different (Commonroad) vehicle models.

    Attributes:
        FordEscort (int): Represents the Ford Escort model.
        BMW320i (int): Represents the BMW 320i model.
        VWVanagon (int): Represents the VW Vanagon model.
        SemiTrailerTruck (int): Represents the Semi-Trailer Truck model.
    """
    FordEscort = 1
    BMW320i = 2
    VWVanagon = 3
    SemiTrailerTruck = 4


@dataclass(kw_only=True)
class StateVector:
    """
    StateVector represents the state of a vehicle in a 2D plane.
    Attributes:
        x (float): x position of the vehicle.
        y (float): y position of the vehicle.
        delta (float): steering angle of the vehicle.
        vel (float): velocity of the vehicle.
        psi (float): yaw angle of the vehicle.
        psi_d (Optional[float]): yaw rate of the vehicle. Default is None.
        beta (Optional[float]): sideslip angle at the vehicle center. Default is None.
        alpha (Optional[float]): hitch angle if KST (Single Track with Trailer) is used. Default is None.
    """
    x: float  # x position
    y: float  # y position
    delta: float  # steering angle
    v: float  # velocity
    psi: float  # yaw angle
    psi_d: Optional[float] = None  # yaw rate
    beta: Optional[float] = None  # sideslip angle at vehicle center
    alpha: Optional[float] = None  # hitch angle if KST (Single Track with Trailer) is used

    @property
    def state_names(self):
        return [i for i, j in self.__dict__.items() if j is not None]

    @property
    def values(self):
        return np.array([j for i, j in self.__dict__.items() if j is not None])


@dataclass(kw_only=True)
class ControlVector:
    """
    ControlVector represents a sequence of control inputs for a vehicle.
    Attributes:
        v_delta (float): steering angle velocity of the vehicle.
        acc (float): velocity acceleration of the vehicle.
        dt (float): time step size of control inputs.
        time_horizon (float): time horizon of control inputs.
        If v_delta and acc area single values, it is used for all steps within the specified time horizon.
        If v_delta and acc are lists, they are used as control inputs for consecutive time steps, and the
            planning horizon is adapted.

    """
    v_delta: Union[List[float], float]  # steering angle velocity
    acc: Union[List[float], float]  # longitudinal acceleration
    dt: float = 0.1  # time step size of control inputs in seconds
    time_horizon: Optional[float] = None  # time horizon of control inputs in seconds
    num_steps: int = field(init=False)  # number of control steps
    u: List[List[float]] = field(init=False)  # control inputs as vector for each time step

    @property
    def control_names(self):
        return ["v_delta", "acc"]

    def __post_init__(self):
        if isinstance(self.v_delta, (list, np.ndarray)) and isinstance(self.acc, (list, np.ndarray)):
            assert len(self.v_delta) == len(self.acc), "Length of v_delta and acc must be equal"
            if self.time_horizon:
                logger.debug("time_horizon is set, but control inputs are lists. \
                       time_horizon will be set to length of control inputs.")
            self.time_horizon = (len(self.v_delta) - 1) * self.dt
        elif isinstance(self.v_delta, (list, np.ndarray)):
            self.acc = [self.acc] * len(self.v_delta)
            if self.time_horizon:
                logger.debug("time_horizon is set, but control inputs are lists. \
                       time_horizon will be set to length of control inputs.")
            self.time_horizon = len(self.v_delta) * self.dt
        elif isinstance(self.acc, (list, np.ndarray)):
            if self.time_horizon:
                logger.debug("time_horizon is set, but control inputs are lists. \
                       time_horizon will be set to length of control inputs.")
            self.v_delta = [self.v_delta] * len(self.acc)
            self.time_horizon = len(self.acc) * self.dt
        else:
            assert self.time_horizon, "time_horizon for control values must be specified \
                if control inputs are not lists"
            self.v_delta = [self.v_delta] * (int(self.time_horizon / self.dt) + 1)
            self.acc = [self.acc] * (int(self.time_horizon / self.dt) + 1)
        self.num_steps = len(self.v_delta)
        self.u = np.array([[self.v_delta[i], self.acc[i]] for i in range(self.num_steps)])
