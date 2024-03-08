from datetime import datetime
from typing import Sequence, List, Dict, Mapping

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.path import Path
from shapely.geometry import Point
from shapely.ops import unary_union

from stonesoup.types.track import Track
from stonesoup.custom.sensor.movable import MovableUAVCamera
from stonesoup.sensor.sensor import Sensor
from stonesoup.types.array import StateVector
from stonesoup.types.numeric import Probability
from stonesoup.types.state import ParticleState, GaussianState
from stonesoup.sensor.action import Action as ssAction


def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def compute_ellipse(cov, pos, nstd=1, **kwargs):

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta,
                    alpha=0.4, **kwargs)
    return ellip.get_path()


def plot_cov_ellipse(cov, pos, nstd=1, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta,
                    alpha=0.4, **kwargs)

    ax.add_artist(ellip)
    return ellip


def _prob_detect_func(prob_detect, fovs):
    """Closure to return the probability of detection function for a given environment scan"""

    # Get the union of all field of views
    fovs_union = unary_union(fovs)
    if fovs_union.geom_type == 'MultiPolygon':
        fovs = [poly for poly in fovs_union]
    else:
        fovs = [fovs_union]

    # Probability of detection nested function
    def prob_detect_func(state):
        for poly in fovs:
            if isinstance(state, ParticleState):
                prob_detect_arr = np.full((len(state),), Probability(0.1))
                path_p = Path(poly.boundary.coords)
                points = state.state_vector[[0, 2], :].T
                inside_points = path_p.contains_points(points)
                prob_detect_arr[inside_points] = prob_detect
                return prob_detect_arr
            else:
                point = Point(state.state_vector[0, 0], state.state_vector[2, 0])
                return prob_detect if poly.contains(point) else Probability(0.1)

    return prob_detect_func