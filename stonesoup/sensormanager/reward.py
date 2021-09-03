# -*- coding: utf-8 -*-

from copy import copy
import numpy as np

from stonesoup.base import Base, Property
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.types.track import Track
from stonesoup.types.hypothesis import SingleHypothesis


class UncertaintyRewardFunction(Base):
    """A reward function which calculates the potential reduction in the uncertainty of track estimates
    if a particular action is taken by a sensor or group of sensors.
    """

    predictor: KalmanPredictor = Property(doc="Predictor used to predict the track to a new state")
    updater: ExtendedKalmanUpdater = Property(doc="Updater used in the reward function to update "
                                                  "the track to the new state.")

    def calculate_reward(self, config, tracks_list, metric_time):
        """Given a configuration of sensors and actions, a metric is calculated for the potential
        reduction in the uncertainty of the tracks that would occur if the sensing configuration
        were used to make an observation. A larger value indicates a greater reduction in
        uncertainty.
        """

        # Reward value
        config_metric = 0

        # Create dictionary of predictions for the tracks in the configuration
        predictions = {track: self.predictor.predict(track[-1],
                                                     timestamp=metric_time)
                       for track in tracks_list}
        # Running updates
        r_updates = dict()

        predicted_sensors = dict()

        # For each sensor in the configuration
        for sensor, actions in config.items():
            #             print(actions)
            predicted_sensor = copy(sensor)
            predicted_sensor.add_actions(actions)
            predicted_sensor.act(metric_time)
            predicted_sensors[sensor] = predicted_sensor

        for sensor, action in config.items():
            # some logic needed here to check if sensor is platform...

            for track in tracks_list:

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

                detections = predicted_sensors[sensor].measure([predicted_track], noise=False)
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
