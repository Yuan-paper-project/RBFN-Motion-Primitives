__author__ = "Marc Kaufed"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

import torch


# Norm functions
def l_norm(x, p=2):
    return torch.norm(x, p=p, dim=-1)


# RBFs
def gaussian(alpha):
    """Gaussian Radial Basis Function"""
    phi = torch.exp(-1*alpha.pow(2))
    return phi


def linear(alpha):
    """Linear Radial Basis Function"""
    phi = alpha
    return phi


def quadratic(alpha):
    """Quadratic Radial Basis Function"""
    phi = alpha.pow(2)
    return phi


def inverse_quadratic(alpha):
    """Inverse Quadratic Radial Basis Function"""
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
    return phi


def multiquadric(alpha):
    """Multiquadric Radial Basis Function"""
    phi = (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi


def inverse_multiquadric(alpha):
    """Inverse Multiquadric Radial Basis Function"""
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi


def spline(alpha):
    """Spline Radial Basis Function"""
    phi = (alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha)))
    return phi


def poisson_one(alpha):
    """Poisson One Radial Basis Function"""
    phi = (alpha - torch.ones_like(alpha)) * torch.exp(-alpha)
    return phi


def poisson_two(alpha):
    """Poisson Two Radial Basis Function"""
    phi = ((alpha - 2*torch.ones_like(alpha)) / 2*torch.ones_like(alpha))\
        * alpha * torch.exp(-alpha)
    return phi


def matern32(alpha):
    """Matern 3/2 Radial Basis Function"""
    phi = (torch.ones_like(alpha) + 3**0.5*alpha)*torch.exp(-3**0.5*alpha)
    return phi


def matern52(alpha):
    """Matern 5/2 Radial Basis Function"""
    phi = (torch.ones_like(alpha) + 5**0.5*alpha + (5/3)
           * alpha.pow(2))*torch.exp(-5**0.5*alpha)
    return phi


def basis_func_dict():
    """
    A helper function that returns a dictionary containing each RBF
    """

    bases = {'gaussian': gaussian,
             'linear': linear,
             'quadratic': quadratic,
             'inverse_quadratic': inverse_quadratic,
             'multiquadric': multiquadric,
             'inverse_multiquadric': inverse_multiquadric,
             'spline': spline,
             'poisson_one': poisson_one,
             'poisson_two': poisson_two,
             'matern32': matern32,
             'matern52': matern52}
    return bases
