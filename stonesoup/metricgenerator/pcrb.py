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


class PCRB(MetricGenerator):
    """
    Computes the (P?) Cramer Rao Bound
    """
    prior: GaussianState = Property(doc="The Gaussian prior")
    transition_model: TransitionModel = Property(doc="The transition model")
    measurement_model: MeasurementModel = Property(doc="The measurement model")
    sensor_locations: StateVectors = Property(doc="The locations of the sensors (currently "
                                                  "assuming sensors are static)")
    clutter_flag: bool = Property(doc="Whether to consider clutter")
    irf_overall: np.ndarray = Property(doc="!!!TODO!!!")
    position_mapping: Sequence[int] = Property(
        default=None,
        doc="Mapping for position coordinates. Default `None`, which uses the measurement model"
            "mapping")
    velocity_mapping: Sequence[int] = Property(
        default=None,
        doc="Mapping for velocity coordinates. Default `None`, in which case velocity RMSE is not "
            "computed")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.position_mapping is None:
            self.position_mapping = self.measurement_model.mapping

    def compute_metric(self, manager, **kwargs):
        pcrb_metrics = []
        for gnd_path in manager.groundtruth_paths:
            pcrb_metric = self._compute_pcrb_single(self.prior, self.transition_model,
                                                    self.measurement_model, gnd_path,
                                                    self.sensor_locations,
                                                    self.clutter_flag, self.irf_overall,
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
                             sensor_locations: StateVectors, clutter_flag: bool, IRF_overall: int,
                             position_mapping: Sequence[int], velocity_mapping: Sequence[int]):

        num_timesteps = len(groundtruth)
        num_sensors = sensor_locations.shape[1]
        ndim_state = transition_model.ndim_state
        ndim_meas = measurement_model.ndim_meas

        # allocate memory
        inverse_J = np.zeros((num_timesteps, ndim_state, ndim_state))
        pos_RMSE = np.zeros(num_timesteps)
        vel_RMSE = np.zeros(num_timesteps)
        IRF = np.zeros((num_sensors, num_timesteps))
        J = np.zeros((num_timesteps + 1, ndim_state, ndim_state))

        for j in range(0, num_sensors):

            ## set IRF values (no clutter)
            if (clutter_flag == 0):
                for i in range(1, num_timesteps):
                    IRF[j][i] = 1.0

            ## use the overall IRF
            elif (clutter_flag == 1):
                for i in range(1, num_timesteps):
                    IRF[j][i] = IRF_overall[0][0]

            ## use the MSC IRF
            else:
                for i in range(1, num_timesteps):
                    IRF[j][i] = IRF_overall[j][i]

        # initialisation
        J[0] = inv(prior.covar)
        inverse_J[0] = inv(J[0])

        # Compute RMSE
        pos_RMSE[0] = cls._compute_pos_RMSE(inverse_J[0], position_mapping)
        if velocity_mapping:
            vel_RMSE[0] = cls._compute_vel_RMSE(inverse_J[0], velocity_mapping)

        # Previous time
        prev_time = groundtruth.states[0].timestamp

        # run Riccati recursion
        for i, state in enumerate(groundtruth.states[1:], 1):
            # Current time
            curr_time = state.timestamp

            # Determine measurement contribution (total_J_Z) - including clutter (if applicable)
            total_J_Z = cls._calculate_J_Z(state, sensor_locations, measurement_model, IRF[:, i])

            # Determine F and Q matrices
            dt = curr_time - prev_time
            F = transition_model.jacobian(state, time_interval=dt)
            Q = transition_model.covar(time_interval=dt)

            # Determine: J[k] = inv(F J^_1 F^T + Q) + J_Z (adjusted for the IRF)
            J[i] = inv(F @ inverse_J[i - 1] @ F.T + Q) + total_J_Z

            # Determine the PCRB
            inverse_J[i] = inv(J[i])

            # Determine the location RMSE
            pos_RMSE[i] = cls._compute_pos_RMSE(inverse_J[i], position_mapping)
            if velocity_mapping:
                vel_RMSE[i] = cls._compute_vel_RMSE(inverse_J[i], velocity_mapping)

            # Update previous time
            prev_time = curr_time

        metric = {}
        metric['track'] = groundtruth
        metric['inverse_J'] = inverse_J
        metric['position_RMSE'] = pos_RMSE
        if velocity_mapping:
            metric['velocity_RMSE'] = vel_RMSE

        return metric

    @staticmethod
    def _calculate_J_Z(state, sensor_locations, measurement_model, IRF):

        # allocate memory / initialisation
        overall_J_Z = np.zeros((state.ndim, state.ndim))

        # inverse of measurement covariance
        measurement_cov_inv = inv(measurement_model.covar())

        num_sensors = sensor_locations.shape[1]
        for i in range(num_sensors):

            sensor_location = sensor_locations[:, i]
            measurement_model_cp = copy(measurement_model)
            if hasattr(measurement_model_cp, 'translation_offset'):
                measurement_model_cp.translation_offset = sensor_location

            H = measurement_model_cp.jacobian(state)

            J_Z = H.T @ measurement_cov_inv @ H

            # increment
            overall_J_Z += IRF[i] * J_Z

        return overall_J_Z

    @staticmethod
    def _compute_pos_RMSE(inverse_J, position_mapping):
        pos_MSE = 0
        for index in position_mapping:
            pos_MSE += inverse_J[index, index]
        return np.sqrt(pos_MSE)

    @staticmethod
    def _compute_vel_RMSE(inverse_J, velocity_mapping):
        vel_MSE = 0
        for index in velocity_mapping:
            vel_MSE += inverse_J[index, index]
        return np.sqrt(vel_MSE)