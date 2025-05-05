__author__ = "Marc Kaufed"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"


def rk4(f, x, u, p, h):
    """
    Runge-Kutta 4th order method for numerical integration.
    :param f: function to integrate
    :param x: state vector
    :param u: control vector
    :param p: parameter vector
    :param h: time step
    :return: updated state vector
    """
    # Calculate the Runge-Kutta coefficients
    k1 = f(x, u, p)
    k2 = f(x + h / 2 * k1, u, p)
    k3 = f(x + h / 2 * k2, u, p)
    k4 = f(x + h * k3, u, p)
    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def euler(f, x, u, p,  h):
    """
    Euler method for numerical integration.
    :param f: function to integrate
    :param x: state vector
    :param u: control vector
    :param p: parameter vector
    :param h: time step
    :return: updated state vector
    """
    # Calculate the Euler step
    return x + h * f(x, u, p)


def heun(f, x, u, p, h):
    """
    Heun method (Improved Euler) for numerical integration.
    :param f: function to integrate
    :param x: state vector
    :param u: control vector
    :param p: parameter vector
    :param h: time step
    :return: updated state vector
    """
    # Calculate the Heun step
    k1 = f(x, u, p)
    k2 = f(x + h * k1, u, p)
    return x + h / 2 * (k1 + k2)
