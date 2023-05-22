from datetime import datetime
from typing import Sequence, List, Dict, Mapping

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.path import Path
from shapely.geometry import Point
from shapely.ops import unary_union

from reactive_isr_core.data import BeliefState, AssetList, GeoLocation, SensorType, ActionList

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


def belief_state_to_tracks(belief: BeliefState) -> Sequence[Track]:
    """Converts a belief state to a set of stonesoup tracks"""
    targets = belief.targets
    tracks = []
    for target_id, target_detection in targets.items():
        state_vector = StateVector([target_detection.location.longitude,
                                    target_detection.velocity.longitude,
                                    target_detection.location.latitude,
                                    target_detection.velocity.latitude,
                                    target_detection.location.altitude,
                                    target_detection.velocity.altitude])
        covariance_matrix = np.zeros((6, 6), dtype=float)
        covariance_matrix[0::2, 0::2] = target_detection.location_error
        covariance_matrix[1::2, 1::2] = target_detection.velocity_error
        metadata = {
            'target_type_confidences': target_detection.target_type_confidences,
        }
        state = GaussianState(state_vector, covariance_matrix,
                              timestamp=target_detection.time)
        track = Track(id=target_id, states=[state], init_metadata=metadata)
        track.exist_prob = Probability(target_detection.confidence)
        tracks.append(track)
    return tracks


def assets_to_sensors(assets: AssetList, region_corners: List[GeoLocation],
                      action_resolutions: Dict[str, float]) -> Sequence[Sensor]:
    """Converts a set of assets to a list of stonesoup sensors"""
    sensors = []
    for asset in assets.assets:
        if SensorType.AERIAL_V_CAMERA in asset.asset_description.sensor_types:
            sensor_position = StateVector([asset.asset_status.location.longitude,
                                           asset.asset_status.location.latitude,
                                           asset.asset_status.location.altitude])
            lim_x = np.sort([loc.longitude for loc in region_corners])
            lim_y = np.sort([loc.latitude for loc in region_corners])
            limits = {'location_x': lim_x, 'location_y': lim_y}
            sensor = MovableUAVCamera(ndim_state=6, mapping=[0, 2, 4],
                                      noise_covar=np.diag([0.05, 0.05, 0.05]),
                                      fov_radius=asset.asset_description.fov_radius,
                                      location_x=sensor_position[0],
                                      location_y=sensor_position[1],
                                      resolutions=action_resolutions,
                                      position=sensor_position,
                                      limits=limits)
            sensor.id = asset.asset_description.id
            sensors.append(sensor)
        else:
            raise NotImplementedError("Only aerial cameras are supported")
    return sensors


def action_list_to_config(assets, action_list: ActionList,
                          region_corners: List[GeoLocation],
                          action_resolutions: Dict[str, float],
                          time: datetime) -> Mapping[Sensor, Sequence[ssAction]]:
    """Converts a reactive_isr_core action list to a stonesoup config"""
    sensors = assets_to_sensors(assets, region_corners, action_resolutions)
    config = {}

    for action in action_list.actions:
        try:
            sensor = next(s for s in sensors if s.id == action.asset_id)
        except StopIteration as exc:
            raise ValueError(f"Asset {action.asset_id} not found") from exc
        location_x, location_y = action.location.longitude, action.location.latitude
        action_generators = sensor.actions(time)
        x_action_gen = next(a for a in action_generators if a.attribute == 'location_x')
        y_action_gen = next(a for a in action_generators if a.attribute == 'location_y')
        x_action = x_action_gen.action_from_value(location_x)
        y_action = y_action_gen.action_from_value(location_y)
        config[sensor] = (x_action, y_action)
    return config