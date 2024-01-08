from abc import ABC
import copy
import datetime
from typing import Mapping, Sequence, Set

import numpy as np

from ..measures import KLDivergence
from ..platform import Platform
from ..sensor.actionable import Actionable
from ..types.detection import TrueDetection
from ..base import Base, Property
from ..predictor.particle import ParticlePredictor
from ..predictor.kalman import KalmanPredictor
from ..updater.kalman import ExtendedKalmanUpdater
from ..types.track import Track
from ..types.hypothesis import SingleHypothesis
from ..sensor.sensor import Sensor
from ..sensor.action import Action
from ..types.prediction import Prediction
from ..updater.particle import ParticleUpdater
from ..resampler.particle import SystematicResampler
from ..types.state import State


class RewardFunction(Base, ABC):
    """
    The reward function base class.

    A reward function is a callable used by a sensor manager to determine the best choice of
    action(s) for a sensor or group of sensors to take. For a given configuration of sensors
    and actions the reward function calculates a metric to evaluate how useful that choice
    of actions would be with a particular objective or objectives in mind.
    The sensor manager algorithm compares this metric for different possible configurations
    and chooses the appropriate sensing configuration to use at that time step.
    """

    def __call__(self, config: Mapping[Sensor, Sequence[Action]], tracks: Set[Track],
                 metric_time: datetime.datetime, *args, **kwargs):
        """
        A method which returns a reward metric based on information about the state of the
        system, sensors and possible actions they can take. This requires a mapping of
        sensors to action(s) to be evaluated by reward function, a set of tracks at given
        time and the time at which the actions would be carried out until.

        Returns
        -------
        : float
            Calculated metric
        """

        raise NotImplementedError


class UncertaintyRewardFunction(RewardFunction):
    """A reward function which calculates the potential reduction in the uncertainty of track
    estimates if a particular action is taken by a sensor or group of sensors.

    Given a configuration of sensors and actions, a metric is calculated for the potential
    reduction in the uncertainty of the tracks that would occur if the sensing configuration
    were used to make an observation. A larger value indicates a greater reduction in
    uncertainty.
    """

    predictor: KalmanPredictor = Property(doc="Predictor used to predict the track to a new state")
    updater: ExtendedKalmanUpdater = Property(doc="Updater used to update "
                                                  "the track to the new state.")
    method_sum: bool = Property(default=True, doc="Determines method of calculating reward."
                                                  "Default calculates sum across all targets."
                                                  "Otherwise calculates mean of all targets.")

    def __call__(self, config: Mapping[Sensor, Sequence[Action]], tracks: Set[Track],
                 metric_time: datetime.datetime, *args, **kwargs):
        """
        For a given configuration of sensors and actions this reward function calculates the
        potential uncertainty reduction of each track by
        computing the difference between the covariance matrix norms of the prediction
        and the posterior assuming a predicted measurement corresponding to that prediction.

        This requires a mapping of sensors to action(s)
        to be evaluated by reward function, a set of tracks at given time and the time at which
        the actions would be carried out until.

        The metric returned is the total potential reduction in uncertainty across all tracks.

        Returns
        -------
        : float
            Metric of uncertainty for given configuration

        """

        # Reward value
        config_metric = 0

        predicted_sensors = set()
        memo = {}
        # For each sensor in the configuration
        for sensor, actions in config.items():
            predicted_sensor = copy.deepcopy(sensor, memo)
            predicted_sensor.add_actions(actions)
            predicted_sensor.act(metric_time)
            if isinstance(sensor, Sensor):
                predicted_sensors.add(predicted_sensor)  # checks if its a sensor
            elif isinstance(sensor, Platform):
                predicted_sensors.update(predicted_sensor.sensors)

        # Create dictionary of predictions for the tracks in the configuration
        predicted_tracks = set()
        for track in tracks:
            predicted_track = copy.copy(track)
            predicted_track.append(self.predictor.predict(predicted_track, timestamp=metric_time))
            predicted_tracks.add(predicted_track)

        for sensor in predicted_sensors:

            # Assumes one detection per track
            detections = {detection.groundtruth_path: detection
                          for detection in sensor.measure(predicted_tracks, noise=False)
                          if isinstance(detection, TrueDetection)}

            for predicted_track, detection in detections.items():
                # Generate hypothesis based on prediction/previous update and detection
                hypothesis = SingleHypothesis(predicted_track.state, detection)

                # Do the update based on this hypothesis and store covariance matrix
                update = self.updater.update(hypothesis)

                previous_cov_norm = np.linalg.norm(predicted_track.covar)
                update_cov_norm = np.linalg.norm(update.covar)

                # Replace prediction with update
                predicted_track.append(update)

                # Calculate metric for the track observation and add to the metric
                # for the configuration
                metric = previous_cov_norm - update_cov_norm
                config_metric += metric

            if self.method_sum is False and len(detections) != 0:

                config_metric /= len(detections)

        # Return value of configuration metric
        return config_metric


class ExpectedKLDivergence(RewardFunction):
    """A reward function that implements the Kullback-Leibler divergence
    for quantifying relative information gain between actions taken by
    a sensor or group of sensors.

    From a configuration of sensors and actions, an expected measurement is
    generated based on the predicted distribution and an action being taken.
    An update is generated based on this measurement. The Kullback-Leibler
    divergence is then calculated between the predicted and updated target
    distribution that resulted from the measurement. A larger divergence
    between these distributions equates to more information gained from
    the action and resulting measurement from that action.
    """

    predictor: ParticlePredictor = Property(default=None,
                                            doc="Predictor used to predict the track to a "
                                                "new state. This reward function is only "
                                                "compatible with :class:`~.ParticlePredictor` "
                                                "types.")
    updater: ParticleUpdater = Property(default=None,
                                        doc="Updater used to update the track to the new state. "
                                            "This reward function is only compatible with "
                                            ":class:`~.ParticleUpdater` types.")
    method_sum: bool = Property(default=True, doc="Determines method of calculating reward."
                                                  "Default calculates sum across all targets."
                                                  "Otherwise calculates mean of all targets.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.KLD = KLDivergence()
        if self.predictor is not None and not isinstance(self.predictor, ParticlePredictor):
            raise NotImplementedError('Only ParticlePredictor types are currently compatible '
                                      'with this reward function')
        if self.updater is not None and not isinstance(self.updater, ParticleUpdater):
            raise NotImplementedError('Only ParticleUpdater types are currently compatible '
                                      'with this reward function')

    def __call__(self, config: Mapping[Sensor, Sequence[Action]], tracks: Set[Track],
                 metric_time: datetime.datetime, *args, **kwargs):
        """
        For a given configuration of sensors and actions this reward function
        calculates the expected Kullback-Leibler divergence of each track. It is
        calculated between the prediction and the posterior assuming an expected update
        based on a predicted measurement.

        This requires a mapping of sensors to action(s) to be evaluated by the
        reward function, a set of tracks at given time and the time at which
        the actions would be carried out until.

        The metric returned is the total expected Kullback-Leibler
        divergence across all tracks.

        Returns
        -------
        : float
            Kullback-Leibler divergence for given configuration

        """

        # Reward value
        kld = 0.

        memo = {}
        predicted_sensors = set()
        # For each actionable in the configuration
        for actionable, actions in config.items():
            # Don't currently have an Actionable base for platforms hence either Platform or Sensor
            if isinstance(actionable, Platform) or isinstance(actionable, Actionable):
                predicted_actionable = copy.deepcopy(actionable, memo)
                predicted_actionable.add_actions(actions)
                predicted_actionable.act(metric_time)
                if isinstance(actionable, Sensor):
                    predicted_sensors.add(predicted_actionable)  # checks if its a sensor
                elif isinstance(actionable, Platform):
                    predicted_sensors.update(predicted_actionable.sensors)

        # Create dictionary of predictions for the tracks in the configuration
        predicted_tracks = set()
        for track in tracks:
            predicted_track = Track()
            if self.predictor:
                predicted_track = copy.copy(track)
                predicted_track.append(self.predictor.predict(track[-1],
                                                              timestamp=metric_time))
            else:
                predicted_track.append(Prediction.from_state(track[-1],
                                                             timestamp=metric_time))

            predicted_tracks.add(predicted_track)

        for sensor in predicted_sensors:
            # Assumes one detection per track

            detections = self._generate_detections(predicted_tracks, sensor)

            for predicted_track, detection_set in detections.items():

                for n, detection in enumerate(detection_set):

                    # if detection:
                    # Generate hypothesis based on prediction/previous update and detection
                    hypothesis = SingleHypothesis(predicted_track, detection)

                    # Do the update based on this hypothesis and store covariance matrix
                    update = self.updater.update(hypothesis)

                    # else:
                    #     update = copy.copy(predicted_track[-1])

                    kld += self.KLD(predicted_track[-1], update)

            if self.method_sum is False and len(detections) != 0:

                kld /= len(detections)

        # Return value of configuration metric
        return kld

    def _generate_detections(self, predicted_tracks, sensor):

        detections = {}
        for predicted_track in predicted_tracks:
            track_detections = set()
            track_detections.update(sensor.measure({State(predicted_track.mean)}, noise=True))

            detections.update({predicted_track: track_detections})

        return detections


class MultiUpdateExpectedKLDivergence(ExpectedKLDivergence):
    """A reward function that implements the Kullback-Leibler divergence
    for quantifying relative information gain between actions taken by
    a sensor or group of sensors.

    From a configuration of sensors and actions, multiple expected measurements per
    track are generated based on the predicted distribution and an action being taken.
    The measurements are generated by resampling the particle state down to a
    subsample with length specified by the user. Updates are generated for each of
    these measurements and the Kullback-Leibler divergence calculated for each
    of them.
    """

    updates_per_track: int = Property(default=2,
                                      doc="Number of measurements to generate from each "
                                          "track prediction. This should be > 1.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.KLD = KLDivergence()
        if self.predictor is not None and not isinstance(self.predictor, ParticlePredictor):
            raise NotImplementedError('Only ParticlePredictor types are currently compatible '
                                      'with this reward function')
        if self.updater is not None and not isinstance(self.updater, ParticleUpdater):
            raise NotImplementedError('Only ParticleUpdater types are currently compatible '
                                      'with this reward function')
        if self.updates_per_track < 2:
            raise ValueError(f'updates_per_track = {self.updates_per_track}. This reward '
                             f'function only accepts >= 2')

    def _generate_detections(self, predicted_tracks, sensor):

        detections = {}

        resampler = SystematicResampler()

        for predicted_track in predicted_tracks:

            measurement_sources = resampler.resample(predicted_track[-1],
                                                     nparts=self.updates_per_track)

            track_detections = set()
            for state in measurement_sources.state_vector:
                track_detections.update(sensor.measure({State(state)}, noise=True))

            detections.update({predicted_track: track_detections})

        return detections
