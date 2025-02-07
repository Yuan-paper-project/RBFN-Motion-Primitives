__author__ = "Marc Kaufed"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"


def acceleration_constraints(velocity, acceleration, p):
    """
    accelerationConstraints - adjusts the acceleration based on acceleration constraints

    Inputs:
        :param acceleration - acceleration in driving direction
        :param velocity - velocity in driving direction
        :params p - longitudinal parameter structure

    Outputs:
        :return acceleration - acceleration in driving direction

    Author: Matthias Althoff
    Written: 15-December-2017
    Last update: ---
    Last revision: ---
    """
    # positive acceleration limit
    a_max = p[0]
    v_switch = p[1]
    v_min = p[2]
    v_max = p[3]
    if velocity > v_switch:
        posLimit = a_max * v_switch / velocity
    else:
        posLimit = a_max

    # acceleration limit reached?
    if (velocity <= v_min and acceleration <= 0) or (velocity >= v_max and acceleration >= 0):
        acceleration = 0
    elif acceleration <= -a_max:
        acceleration = -a_max
    elif acceleration >= posLimit:
        acceleration = posLimit

    return acceleration


def jerk_dot_constraints(jerk_dot, jerk, p):
    """
    input constraints for jerk_dot: adjusts jerk_dot if jerk limit or input bounds are reached
    """
    if (jerk_dot < 0. and jerk <= -p.j_max) or (jerk_dot > 0. and jerk >= p.j_max):
        # jerk limit reached
        jerk_dot = 0.
    elif abs(jerk_dot) >= p.j_dot_max:
        # input bounds reached
        jerk_dot = p.j_dot_max
    return jerk_dot
