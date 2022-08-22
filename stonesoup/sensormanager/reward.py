from abc import ABC
import copy
import datetime
from typing import Mapping, Sequence, Set

import numpy as np

from ..base import Base, Property
from ..predictor.kalman import KalmanPredictor
from ..updater.kalman import ExtendedKalmanUpdater
from ..types.track import Track
from ..types.hypothesis import SingleHypothesis
from ..sensor.sensor import Sensor
from ..sensor.action import Action


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
    """A reward function which calculates the potential reduction in the uncertainty of track estimates
    if a particular action is taken by a sensor or group of sensors.

    Given a configuration of sensors and actions, a metric is calculated for the potential
    reduction in the uncertainty of the tracks that would occur if the sensing configuration
    were used to make an observation. A larger value indicates a greater reduction in
    uncertainty.
    """

    predictor: KalmanPredictor = Property(doc="Predictor used to predict the track to a new state")
    updater: ExtendedKalmanUpdater = Property(doc="Updater used to update "
                                                  "the track to the new state.")

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

        predicted_sensors = list()
        memo = {}
        # For each sensor in the configuration
        for sensor, actions in config.items():
            predicted_sensor = copy.deepcopy(sensor, memo)
            predicted_sensor.add_actions(actions)
            predicted_sensor.act(metric_time)
            if isinstance(sensor, Sensor):
                predicted_sensors.append(predicted_sensor)  # checks if its a sensor

        # Create dictionary of predictions for the tracks in the configuration
        predicted_tracks = set()
        for track in tracks:
            predicted_track = copy.copy(track)
            predicted_track.append(self.predictor.predict(predicted_track, timestamp=metric_time))
            predicted_tracks.add(predicted_track)

        for sensor in predicted_sensors:

            # Assumes one detection per track
            detections = {detection.groundtruth_path: detection
                          for detection in sensor.measure(predicted_tracks, noise=False)}

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

        # Return value of configuration metric
        return config_metric
