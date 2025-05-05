__author__ = "Marc Kaufed"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"
import sys
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import matplotlib.transforms as transforms
from commonroad.visualization.icons import draw_car_icon
from matplotlib.legend_handler import HandlerPatch, HandlerLine2D


class EgoPatchHandler(HandlerPatch):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        return draw_car_icon(
            10, 4, np.pi, vehicle_length=25, vehicle_width=10, vehicle_color="#E37222", edgecolor="#9C4100"
        )


class ObsPatchHandler(HandlerPatch):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        return draw_car_icon(
            10, 4, np.pi, vehicle_length=25, vehicle_width=10, vehicle_color="#2266e3", edgecolor="#003359"
        )


class TrajPatchHandler(HandlerLine2D):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        line = Line2D(
            [xdescent + width * 0.1, xdescent + width * 0.9],
            [ydescent + height / 2, ydescent + height / 2],
            color=orig_handle.get_color(),
            linestyle=orig_handle.get_linestyle(),
            linewidth=orig_handle.get_linewidth(),
        )
        marker1 = Line2D(
            [xdescent + width * 0.25],
            [ydescent + height / 2],
            color=orig_handle.get_color(),
            marker=orig_handle.get_marker(),
            markersize=orig_handle.get_markersize(),
            linestyle="None",
        )
        marker2 = Line2D(
            [xdescent + width * 0.75],
            [ydescent + height / 2],
            color=orig_handle.get_color(),
            marker=orig_handle.get_marker(),
            markersize=orig_handle.get_markersize(),
            linestyle="None",
        )
        return [line, marker1, marker2]


def confidence_ellipse(mu, cov, ax, n_std=3.0, facecolor="red", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """

    mu_x = mu[0]
    mu_y = mu[1]

    pearson = cov[0][1] / (np.sqrt(cov[0][0] * cov[1][1]) + sys.float_info.epsilon)
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        alpha=0.2,
        zorder=55,
        **kwargs
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0][0]) * n_std

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1][1]) * n_std

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mu_x, mu_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def draw_with_uncertainty(fut_pos_list, fut_cov_list, ax):

    for i, fut_pos in enumerate(fut_pos_list):
        for j, pos in enumerate(fut_pos):
            confidence_ellipse(
                pos, fut_cov_list[i][j], ax, n_std=1.0, facecolor="yellow"
            )
        for j, pos in enumerate(fut_pos):
            confidence_ellipse(
                pos, fut_cov_list[i][j], ax, n_std=0.5, facecolor="orange"
            )
        for j, pos in enumerate(fut_pos):
            confidence_ellipse(pos, fut_cov_list[i][j], ax, n_std=0.2, facecolor="red")


def draw_uncertain_predictions(prediction_dict, ax):
    """Draw predictions and visualize uncertainties with heat maps.

    Args:
        prediction_dict ([dict]): [prediction dicts with key obstacle id and value pos_list and cov_list]
        ax ([type]): [matpllotlib.ax to plot in]
    """

    prediction_plot_list = list(prediction_dict.values())[:10]
    fut_pos_list = [
        prediction_plot_list[i]["pos_list"][:20][:]
        for i in range(len(prediction_plot_list))
    ]

    fut_cov_list = [
        prediction_plot_list[i]["cov_list"][:20][:]
        for i in range(len(prediction_plot_list))
    ]
    draw_with_uncertainty(fut_pos_list, fut_cov_list, ax)
