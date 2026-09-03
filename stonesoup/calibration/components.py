import copy
import datetime
from abc import abstractmethod
import numpy as np
from scipy.linalg import block_diag
from collections import defaultdict
from typing import Any
from typing import DefaultDict
from stonesoup.base import Base, Property
from stonesoup.dataassociator import DataAssociator

# from stonesoup.models.measurement.bias import BiasModelWrapper
from .model import BiasModelWrapper
from stonesoup.models.measurement.nonlinear import (
    CombinedReversibleGaussianMeasurementModel,
)
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.types.update import Update
from stonesoup.updater import Updater
from stonesoup.updater.kalman import UnscentedKalmanUpdater
from stonesoup.models.measurement import MeasurementModel
from stonesoup.models.transition import TransitionModel
from stonesoup.base import Property
from stonesoup.functions import jacobian as compute_jac
from stonesoup.types.state import State, StateVectors
from stonesoup.feeder.base import DetectionFeeder
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.tracker.simple import MultiTargetTracker
from stonesoup.types.state import GaussianState


class GaussianBiasUpdater(Updater):
    """Updater that jointly estimates a bias alongside target states.

    Maintains a separate Gaussian bias state and integrates it with target predictions
    to perform joint prediction and update steps. Uses a provided non-linear `updater`
    (defaults to an Unscented Kalman updater) and a `bias_model_wrapper` to build
    joint measurement models for bias estimation.

    Note that this assumes that all measurements/hypotheses are updating a common
    bias i.e. all measurements from the same sensor.
    """

    measurement_model = None
    bias_predictor: KalmanPredictor = Property(doc="Predictor for bias", default=None)
    # TODO: pass through dictionary that keys sensor to bias model wrapper. Not all will be the same bias state space.
    bias_model_wrapper: BiasModelWrapper = Property()
    updater: Updater = Property(
        default=None,
        doc="Updater for bias and joint states. Must support non-linear models. "
        "Default `None` will create UKF instance.",
    )
    max_bias: list[float] = Property(default=None, doc="Max bias ± from 0 allowed")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.updater is None:
            self.updater = UnscentedKalmanUpdater(None)

    def predict_measurement(
        self,
        predicted_state,
        measurement_model=None,
        measurement_noise=True,
        bias_state=None,
        **kwargs,
    ):
        if bias_state is None:
            return self.updater.predict_measurement(
                predicted_state, measurement_model, measurement_noise, **kwargs
            )
        ndim_bias = bias_state.ndim
        if bias_state.timestamp is None:
            pred_bias_state = copy.copy(bias_state)
            pred_bias_state.timestamp = predicted_state.timestamp
        else:
            pred_bias_state = self.bias_predictor.predict(
                bias_state, timestamp=predicted_state.timestamp
            )

        combined_pred = GaussianState(
            np.vstack(
                [predicted_state.state_vector, pred_bias_state.state_vector]
            ).view(StateVector),
            block_diag(*[predicted_state.covar, pred_bias_state.covar]).view(
                CovarianceMatrix
            ),
            timestamp=predicted_state.timestamp,
        )

        bias_measurement_model = self.bias_model_wrapper(
            ndim_state=combined_pred.state_vector.shape[0],
            measurement_model=measurement_model,
            state_mapping=list(range(predicted_state.ndim)),
            bias_mapping=list(range(-ndim_bias, 0)),
        )
        return self.updater.predict_measurement(
            combined_pred, bias_measurement_model, measurement_noise, **kwargs
        )

    def update(
        self,
        hypotheses: list[SingleHypothesis],
        bias_state: GaussianState | None = None,
        **kwargs,
    ) -> list[Update]:
        if any(not hyp for hyp in hypotheses):
            raise ValueError("Must provide only non-missed detection hypotheses")

        if bias_state is None:
            return [self.updater.update(hypothesis) for hypothesis in hypotheses], None

        ndim_bias = bias_state.ndim
        pred_time = max(hypothesis.prediction.timestamp for hypothesis in hypotheses)
        if bias_state.timestamp is None:
            bias_state = copy.copy(bias_state)
            bias_state.timestamp = pred_time
        else:
            bias_state = self.bias_predictor.predict(bias_state, timestamp=pred_time)

        # Create joint state
        states = [hypothesis.prediction for hypothesis in hypotheses]
        states.append(bias_state)
        combined_pred = GaussianState(
            np.vstack([state.state_vector for state in states]).view(StateVector),
            block_diag(*[state.covar for state in states]).view(CovarianceMatrix),
            timestamp=pred_time,
        )

        # Create joint measurement
        offset = 0
        models = []
        for prediction, measurement in (
            (hypothesis.prediction, hypothesis.measurement) for hypothesis in hypotheses
        ):
            models.append(
                self.bias_model_wrapper(
                    ndim_state=combined_pred.state_vector.shape[0],
                    measurement_model=measurement.measurement_model,
                    state_mapping=[offset + n for n in range(prediction.ndim)],
                    bias_mapping=list(range(-ndim_bias, 0)),
                )
            )
            offset += prediction.ndim
        combined_meas = Detection(
            np.vstack(
                [hypothesis.measurement.state_vector for hypothesis in hypotheses]
            ),
            timestamp=pred_time,
            measurement_model=CombinedReversibleGaussianMeasurementModel(models),
        )

        # Update bias
        update = self.updater.update(
            SingleHypothesis(combined_pred, combined_meas), **kwargs
        )
        bias_state.state_vector = update.state_vector[-ndim_bias:, :]
        if self.max_bias is not None:
            bias_state.state_vector = np.min(
                [abs(bias_state.state_vector), self.max_bias], axis=0
            ) * np.sign(bias_state.state_vector)
        bias_state.covar = update.covar[-ndim_bias:, -ndim_bias:]

        # Create update states
        offset = 0
        updates = []
        for hypothesis in hypotheses:
            update_slice = slice(offset, offset + hypothesis.prediction.ndim)
            updates.append(
                Update.from_state(
                    hypothesis.prediction,
                    state_vector=update.state_vector[update_slice, :],
                    covar=update.covar[update_slice, update_slice],
                    timestamp=hypothesis.prediction.timestamp,
                    hypothesis=hypothesis,
                )
            )
            offset += hypothesis.prediction.ndim
        return updates, bias_state


class TrackSelector(Base):
    """
    Track selector base type

    A `TrackSelector` selects a set of tracks by a specified set of restraints.
    """

    @abstractmethod
    def __call__(
        self,
        tracks: set[Track],
        time: datetime.datetime,
        sensor_id: Any,
        shared_bias_track_store: dict[Any, Track] = {},
    ) -> set[Track]:
        """
        Args:
            tracks (set[Tracks]): Set of tracks to filter down.
            time (datetime): Time that track selecting is taking place.
            sensor_id (Any): ID of the sensor that provided the detections at this timestep. It is assumed that
                a detector yields detections on a per sensor basis when calibrating.
            track_store (dict[Any, Track]): Dictionary of sensor's bias tracks, keyed by the sensor's id.

        Returns:

        """
        raise NotImplementedError


class DummyTrackSelector(TrackSelector):
    """
    Will return all tracks present.
    """

    def __call__(self, tracks, time, sensor_id, shared_bias_track_store={}):
        return tracks


class TimeTrackSelector(TrackSelector):
    """
    Will return a track if a sensor with an ID in `fixed_sensor_ids` has seen the track within the
    specified time window
    """

    window_length: float = Property(
        doc="Number of seconds since fixed sensor has observed a track."
    )
    fixed_sensor_ids: list[Any] = Property(
        doc="List of IDs of sensors whose coordinate system other sensors will be calibrated to."
    )

    def __call__(
        self,
        tracks: set[Track],
        time: datetime.datetime,
        sensor_id: Any,
        bias_track_store: dict[Any, Track] = {},
    ) -> set[Track]:
        calibration_tracks = set()
        for track in tracks:
            states = track.states[
                -len(track[(time - datetime.timedelta(seconds=self.window_length)) :]) :
            ]
            # all sensor ids that have viewed the track in recent time window
            detections = {
                state.hypothesis.measurement.metadata["sensorId"]
                for state in states
                if isinstance(state, Update)
            }
            # if an id of a fixed sensor in recent detections associated to track
            if set(self.fixed_sensor_ids) & detections:
                calibration_tracks.add(track)
        return calibration_tracks


class BiasUncertaintyTrackSelector(TimeTrackSelector):
    """
    A sensor will be added to the `fixed_sensor_ids` list if the trace of its bias estimate's
    covariance matrix has dropped below the specified threshold.

    A sensor cannot calibrate using a track if it is the only fixed sensor contributing to the track
    in the specified time window.

    TODO: add mapping to allow weighting of components in covariance matrix (radians^2 vs meters^2 etc)
    TODO: and dictionary to bias_sensor_track of the (main?) sensors used to calibrate from. This can be used to
        not calibrate sensors higher in the hierarchy from sensors calibrated from them.
    """

    covariance_trace_threshold: float = Property(
        doc="Any sensor with a current bias estimate uncertainty with trace below this threshold will"
        "be considered a 'fixed sensor', and will be used to calibrate other sensors."
    )

    def __call__(
        self,
        tracks: set[Track],
        time: datetime.datetime,
        sensor_id: Any,
        bias_track_store: dict[Any, Track] = {},
    ) -> set[Track]:
        fixed_sensors = set(self.fixed_sensor_ids)
        for biased_sensor_id, bias_track in bias_track_store.items():
            if biased_sensor_id in fixed_sensors or bias_track is None:
                continue
            if np.trace(bias_track[-1].covar) <= self.covariance_trace_threshold:
                fixed_sensors.add(biased_sensor_id)

        calibration_tracks = set()
        for track in tracks:
            states = track.states[
                -len(track[(time - datetime.timedelta(seconds=self.window_length)) :]) :
            ]
            # all sensor ids that have viewed the track in recent time window
            detections = {
                state.hypothesis.measurement.metadata["sensorId"]
                for state in states
                if isinstance(state, Update)
            }
            # check for an id of a fixed sensor (that is not the current sensor_id) in recent track states
            fixed_sensors_viewing_track = fixed_sensors & detections
            fixed_sensors_viewing_track -= {sensor_id}
            if fixed_sensors_viewing_track:
                calibration_tracks.add(track)
        return calibration_tracks


class BiasMultiTargetTracker(MultiTargetTracker):
    """
    Bias aware `MultiTargetTracker` that updates sensor's bias estimates with calibration tracks filtered down
    by a `TrackSelector`.
    """

    bias_updater: GaussianBiasUpdater = Property(doc="Bias updater.")
    bias_data_associator: DataAssociator = Property(
        doc="Data associator that considers bias when calculating the predicted measurement."
        "Its hypothesiser must use a bias updater."
    )
    calibration_track_selector: TrackSelector = Property(
        doc="Track selector responsible for deciding which tracks to calibrate from."
    )
    bias_track_store: defaultdict[Any, Track[GaussianState]] = Property(
        doc="Shared dictionary between the tracker, feeders and updater to enable accessing of all sensor's bias estimates, keyed by sensor id."
    )

    def __next__(self, *args, **kwargs):
        time, detections = next(self.detector_iter)

        associated_detections = set()

        if len(self._tracks) == 0:
            self._tracks |= self.initiator.initiate(detections, time)
            return time, self.tracks
        if len(detections) > 0:
            sensor_id = next(detection.metadata["sensorId"] for detection in detections)
            if sensor_id is None:
                raise ValueError("sensor_id has not been set on detection.")
            try:
                bias_track = self.bias_track_store[sensor_id]
            except KeyError:
                bias_track = None
        else:
            bias_track = None

        if bias_track is not None:
            calibration_tracks = self.calibration_track_selector(
                self._tracks, time, sensor_id, self.bias_track_store
            )
            bias_hypotheses = self.bias_data_associator.associate(
                self._tracks, detections, time, bias_state=bias_track.state
            )
            update_hypotheses = {
                track: hypothesis
                for track, hypothesis in bias_hypotheses.items()
                if hypothesis.measurement and track in calibration_tracks
            }

            if len(update_hypotheses) > 0:
                updates, bias_state = self.bias_updater.update(
                    [hypothesis for track, hypothesis in update_hypotheses.items()],
                    bias_state=bias_track.state,
                )
                bias_track.append(bias_state)
            else:
                updates = list()

            # update calibration tracks.
            for track, update in zip(
                (t for t, h in update_hypotheses.items() if h), updates
            ):
                associated_detections.add(update.hypothesis.measurement)
                track.append(update)

            for track, hypothesis in bias_hypotheses.items():
                if track in update_hypotheses.keys():
                    continue
                if hypothesis.measurement:
                    associated_detections.add(hypothesis.measurement)
                    # update non-calibration tracks. Don't update internal estimate of the bias.
                    updates, _ = self.bias_updater.update(
                        [hypothesis], bias_state=bias_track.state
                    )
                    track.append(updates[0])
                else:
                    track.append(hypothesis.prediction)

        else:
            # Standard track update if no bias applied i.e. unbiased sensors
            associations = self.data_associator.associate(
                self._tracks, detections, time
            )
            for track, hypothesis in associations.items():
                if hypothesis.measurement:
                    post = self.updater.update(hypothesis)
                    track.append(post)
                    associated_detections.add(hypothesis.measurement)
                else:
                    track.append(hypothesis.prediction)

        self._tracks -= self.deleter.delete_tracks(self.tracks)
        self._tracks |= self.initiator.initiate(
            detections - associated_detections, time
        )

        return time, self.tracks
