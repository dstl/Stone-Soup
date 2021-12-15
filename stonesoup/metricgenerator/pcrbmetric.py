import numpy as np
from numpy.linalg import inv
from copy import copy
from typing import Sequence

from .base import MetricGenerator
from ..base import Property
from ..types.state import GaussianState
from ..types.groundtruth import GroundTruthPath
from ..types.array import StateVectors
from ..models.transition import TransitionModel
from ..models.measurement import MeasurementModel
from ..types.metric import TimeRangeMetric
from ..types.time import TimeRange


class PCRBMetric(MetricGenerator):
    """
    Computes the Posterior Cramer Rao Bound (PCRB) [1] for a given ground truth prior. PCRB
    provides a MSE bound on the performance of unbiased filtering algorithms.

    Reference:
        [1] M. L. Hernandez and A. Farina, "Posterior Cramér-Rao Bound for Target Tracking in the
        Presence of Multipath," 2018 21st International Conference on Information Fusion (FUSION),
        2018, pp. 151-158, doi: 10.23919/ICIF.2018.8455619.
    """
    prior: GaussianState = Property(doc="The prior used to initiate the track")
    transition_model: TransitionModel = Property(
        doc="The transition model used to propagate the track's state")
    measurement_model: MeasurementModel = Property(
        doc="The measurement model that projects a track into measurement space (and vice versa")
    sensor_locations: StateVectors = Property(
        doc="The locations of the sensors (currently assuming sensors are static)")
    position_mapping: Sequence[int] = Property(
        default=None,
        doc="Mapping for position coordinates. Default `None`, which uses the measurement model"
            "mapping")
    velocity_mapping: Sequence[int] = Property(
        default=None,
        doc="Mapping for velocity coordinates. Default `None`, in which case velocity RMSE is not "
            "computed")
    irf: float = Property(doc="Information reduction factor. Default is 1", default=1.)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.position_mapping is None:
            self.position_mapping = self.measurement_model.mapping

    def compute_metric(self, manager, **kwargs):
        pcrb_metrics = []
        for gnd_path in manager.groundtruth_paths:
            pcrb_metric = self._compute_pcrb_single(self.prior, self.transition_model,
                                                    self.measurement_model, gnd_path,
                                                    self.sensor_locations, self.irf,
                                                    self.position_mapping, self.velocity_mapping)
            time_range = TimeRange(gnd_path.states[0].timestamp, gnd_path.timestamp)
            pcrb_metrics.append(TimeRangeMetric(title='PCRB Metrics',
                                                value=pcrb_metric,
                                                time_range=time_range,
                                                generator=self))
        return pcrb_metrics

    @classmethod
    def _compute_pcrb_single(cls, prior: GaussianState, transition_model: TransitionModel,
                             measurement_model: MeasurementModel, groundtruth: GroundTruthPath,
                             sensor_locations: StateVectors, irf_overall: float,
                             position_mapping: Sequence[int], velocity_mapping: Sequence[int]):
        """ Compute the PCRB for a single Ground truth path

        Parameters
        ----------
        prior: GaussianState
            The prior used to initiate the track
        transition_model: TransitionModel
            The transition model used to propagate the track's state
        measurement_model: MeasurementModel
            The measurement model that projects a track into measurement space (and vice versa)
        groundtruth: GroundTruthPath
            The ground truth path
        sensor_locations: StateVectors
            The locations of the sensors (currently assuming sensors are static)
        irf_overall: float
            Information reduction factor
        position_mapping: list of int
            Mapping for position coordinates. Default `None`, which uses the measurement model
            mapping
        velocity_mapping: list of int
            Mapping for velocity coordinates. Default `None`, in which case velocity RMSE is not
            computed

        Returns
        -------
        dict
            A dictionary with keys:
            - `track`: The groundtruth track
            - `inverse_j`: A matrix of shape (`ndim_state`, `ndim_state`)
            - `position_RMSE`: The MSE bound on the positional RMSE
            - `velocity_RMSE`: The MSE bound on the velocity RMSE (only provided if \
                `velocity_mapping` is not None)
        """

        num_timesteps = len(groundtruth)
        num_sensors = sensor_locations.shape[1]
        ndim_state = transition_model.ndim_state

        # allocate memory
        inverse_j = np.zeros((num_timesteps, ndim_state, ndim_state))
        pos_rmse = np.zeros(num_timesteps)
        vel_rmse = np.zeros(num_timesteps)
        irf = np.ones((num_sensors, num_timesteps))*irf_overall
        j = np.zeros((num_timesteps + 1, ndim_state, ndim_state))

        # initialisation
        j[0] = inv(prior.covar)
        inverse_j[0] = inv(j[0])

        # Compute RMSE
        pos_rmse[0] = cls._compute_pos_rmse(inverse_j[0], position_mapping)
        if velocity_mapping:
            vel_rmse[0] = cls._compute_vel_rmse(inverse_j[0], velocity_mapping)

        # Previous time
        prev_time = groundtruth.states[0].timestamp

        # run Riccati recursion
        for i, state in enumerate(groundtruth.states[1:], 1):
            # Current time
            curr_time = state.timestamp

            # Determine measurement contribution (total_j_z) - including clutter (if applicable)
            total_j_z = cls._calculate_j_z(state, sensor_locations, measurement_model, irf[:, i])

            # Determine F and Q matrices
            dt = curr_time - prev_time
            f_matrix = transition_model.jacobian(state, time_interval=dt)
            q_matrix = transition_model.covar(time_interval=dt)

            # Determine: j[k] = inv(F j^_1 F^T + Q) + j_z (adjusted for the irf)
            j[i] = inv(f_matrix @ inverse_j[i - 1] @ f_matrix.T + q_matrix) + total_j_z

            # Determine the PCRB
            inverse_j[i] = inv(j[i])

            # Determine the location RMSE
            pos_rmse[i] = cls._compute_pos_rmse(inverse_j[i], position_mapping)
            if velocity_mapping:
                vel_rmse[i] = cls._compute_vel_rmse(inverse_j[i], velocity_mapping)

            # Update previous time
            prev_time = curr_time

        metric = {'track': groundtruth,
                  'inverse_j': inverse_j,
                  'position_RMSE': pos_rmse}
        if velocity_mapping:
            metric['velocity_RMSE'] = vel_rmse

        return metric

    @staticmethod
    def _calculate_j_z(state, sensor_locations, measurement_model, irf):

        # allocate memory / initialisation
        overall_j_z = np.zeros((state.ndim, state.ndim))

        # inverse of measurement covariance
        measurement_cov_inv = inv(measurement_model.covar())

        num_sensors = sensor_locations.shape[1]
        for i in range(num_sensors):

            sensor_location = sensor_locations[:, i]
            measurement_model_cp = copy(measurement_model)
            if hasattr(measurement_model_cp, 'translation_offset'):
                measurement_model_cp.translation_offset = sensor_location

            h_matrix = measurement_model_cp.jacobian(state)

            j_z = h_matrix.T @ measurement_cov_inv @ h_matrix

            # increment
            overall_j_z += irf[i] * j_z

        return overall_j_z

    @staticmethod
    def _compute_pos_rmse(inverse_j, position_mapping):
        pos_mse = 0
        for index in position_mapping:
            pos_mse += inverse_j[index, index]
        return np.sqrt(pos_mse)

    @staticmethod
    def _compute_vel_rmse(inverse_j, velocity_mapping):
        vel_mse = 0
        for index in velocity_mapping:
            vel_mse += inverse_j[index, index]
        return np.sqrt(vel_mse)
