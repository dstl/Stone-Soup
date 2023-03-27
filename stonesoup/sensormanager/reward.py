from copy import deepcopy
import numpy as np
import datetime

from abc import ABC
from typing import Mapping, Sequence, Set
from stonesoup.base import Base, Property
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.types.track import Track
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.sensor.sensor import Sensor
from stonesoup.sensor.action import Action
from stonesoup.platform import Platform


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

        # Create dictionary of predictions for the tracks in the configuration
        predictions = {track: self.predictor.predict(track[-1],
                                                     timestamp=metric_time)
                       for track in tracks}
        # Running updates
        r_updates = dict()

        predicted_sensors = list()
        predicted_platforms = list()

        memo = {}

        # For each sensor (or platform) in the configuration
        for actionable, actions in config.items():
            predicted_actionable = deepcopy(actionable, memo)
            predicted_actionable.add_actions(actions)
            predicted_actionable.act(metric_time)
            if isinstance(actionable, Sensor):
                predicted_sensors.append(predicted_actionable)
            elif isinstance(actionable, Platform):
                predicted_platforms.append(predicted_actionable)

        for sensor in predicted_sensors:

            for track in tracks:

                # If the track is selected by a sensor for the first time -
                # 'previous' is the prediction
                # If the track has already been selected by a sensor -
                # 'previous' is the most recent update
                if track not in r_updates:
                    previous = predictions[track]
                else:
                    previous = r_updates[track]

                previous_cov_norm = np.linalg.norm(previous.covar)

                predicted_track = Track(previous, init_metadata=dict(Length=3, Width=1))

                detections = sensor.measure([predicted_track], noise=False)
                if not detections:
                    continue

                detection = detections.pop()  # assumes one detection

                # Generate hypothesis based on prediction/previous update and detection
                hypothesis = SingleHypothesis(previous, detection)

                # Do the update based on this hypothesis and store covariance matrix
                update = self.updater.update(hypothesis)
                update_cov_norm = np.linalg.norm(update.covar)

                # Replace prediction in dictionary with update
                r_updates[track] = update

                # Calculate metric for the track observation and add to the metric
                # for the configuration
                metric = previous_cov_norm - update_cov_norm
                config_metric += metric

        # Return value of configuration metric
        return config_metric
