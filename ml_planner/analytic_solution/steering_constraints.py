__author__ = "Marc Kaufeld"
__copyright__ = "TUM Professorship Autonomous Vehicle Systems"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"


def steering_constraints(steering_angle, steering_velocity, p):
    """
    steering_constraints - adjusts the steering velocity based on steering

    Inputs:
        :param steering_angle - steering angle
        :param steering_velocity - steering velocity
        :params p - steering parameter structure

    Outputs:
        :return steering_velocity - steering velocity

    Author: Matthias Althoff
    Written: 15-December-2017
    Last update: ---
    Last revision: ---
    """
    # steering limit reached?
    min_delta = p[0]
    max_delta = p[1]
    min_v_delta = p[2]
    max_v_delta = p[3]
    if ((steering_angle <= min_delta and steering_velocity <= 0)
            or (steering_angle >= max_delta and steering_velocity >= 0)):
        steering_velocity = 0
    elif steering_velocity <= min_v_delta:
        steering_velocity = min_v_delta
    elif steering_velocity >= max_v_delta:
        steering_velocity = max_v_delta

    return steering_velocity


def kappa_dot_dot_constraints(kappa_dot_dot, kappa_dot, p):
    """
    input constraints for kappa_dot_dot: adjusts kappa_dot_dot if kappa_dot limit (i.e., maximum curvature rate)
    or input bounds are reached
    """
    if (kappa_dot < -p.kappa_dot_max and kappa_dot_dot < 0.) \
            or (kappa_dot > p.kappa_dot_max and kappa_dot_dot > 0.):
        # kappa_dot limit reached
        kappa_dot_dot = 0.
    elif abs(kappa_dot_dot) >= p.kappa_dot_dot_max:
        # input bounds reached
        kappa_dot_dot = p.kappa_dot_dot_max
    return kappa_dot_dot
