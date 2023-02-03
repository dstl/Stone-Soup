import copy
import datetime
from typing import Mapping, Sequence, Set, List, Any
import itertools as it

import numpy as np
from matplotlib.path import Path
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from reactive_isr_core.data import TaskType
from stonesoup.base import Property
from stonesoup.custom.functions import calculate_num_targets_dist, geodesic_point_buffer
from stonesoup.custom.tracker import SMCPHD_JIPDA
from stonesoup.functions import gm_reduce_single
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.sensor.action import Action
from stonesoup.sensor.sensor import Sensor
from stonesoup.sensormanager.reward import RewardFunction
from stonesoup.tracker import Tracker
from stonesoup.types.array import StateVectors
from stonesoup.types.detection import TrueDetection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.numeric import Probability
from stonesoup.types.state import ParticleState
from stonesoup.types.track import Track
from stonesoup.types.update import GaussianStateUpdate
from stonesoup.updater.kalman import ExtendedKalmanUpdater


class RolloutUncertaintyRewardFunction(RewardFunction):
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
    timesteps: int = Property(doc="Number of timesteps to rollout")
    num_samples: int = Property(doc="Number of samples to take for each timestep", default=30)
    interval: datetime.timedelta = Property(doc="Interval between timesteps",
                                            default=datetime.timedelta(seconds=1))

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
        end_time = metric_time + datetime.timedelta(seconds=self.timesteps)
        config_metric = self._rollout(config, tracks, metric_time, end_time)

        # Return value of configuration metric
        return config_metric

    def _rollout(self, config: Mapping[Sensor, Sequence[Action]], tracks: Set[Track],
                 timestamp: datetime.datetime, end_time: datetime.datetime):
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
            predicted_sensor.act(timestamp)
            if isinstance(sensor, Sensor):
                predicted_sensors.append(predicted_sensor)  # checks if its a sensor

        # Create dictionary of predictions for the tracks in the configuration
        predicted_tracks = set()
        for track in tracks:
            predicted_track = copy.copy(track)
            predicted_track.append(self.predictor.predict(predicted_track, timestamp=timestamp))
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

        if timestamp == end_time:
            return config_metric

        timestamp = timestamp + datetime.timedelta(seconds=1)

        all_action_choices = dict()
        for sensor in predicted_sensors:
            # get action 'generator(s)'
            action_generators = sensor.actions(timestamp)
            # list possible action combinations for the sensor
            action_choices = list(it.product(*action_generators))
            # dictionary of sensors: list(action combinations)
            all_action_choices[sensor] = action_choices

        configs = list({sensor: action
                        for sensor, action in zip(all_action_choices.keys(), actionconfig)}
                       for actionconfig in it.product(*all_action_choices.values()))

        idx = np.random.choice(len(configs), self.num_samples)
        configs = [configs[i] for i in idx]

        rewards = [self._rollout(config, tracks, timestamp, end_time) for config in configs]
        config_metric += np.max(rewards)

        return config_metric


class RolloutPriorityRewardFunction(RewardFunction):
    """A reward function which calculates the potential reduction in the uncertainty of track estimates
    if a particular action is taken by a sensor or group of sensors.

    Given a configuration of sensors and actions, a metric is calculated for the potential
    reduction in the uncertainty of the tracks that would occur if the sensing configuration
    were used to make an observation. A larger value indicates a greater reduction in
    uncertainty.
    """

    tracker: Tracker = Property(doc="Tracker used to track the tracks")
    timesteps: int = Property(doc="Number of timesteps to rollout")
    num_samples: int = Property(doc="Number of samples to take for each timestep", default=30)
    interval: datetime.timedelta = Property(doc="Interval between timesteps",
                                            default=datetime.timedelta(seconds=1))
    rfis: List[Any] = Property(doc="List of reward functions to use for prioritisation",
                               default=None)
    prob_survive: Probability = Property(doc="Probability of survival", default=Probability(0.99))
    use_variance: bool = Property(doc="Use variance in prioritisation", default=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.rfis is None:
            self.rfis = []

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
        end_time = metric_time + datetime.timedelta(seconds=self.timesteps)

        config_metric = self._rollout(config, tracks, metric_time, end_time)

        # Return value of configuration metric
        return config_metric

    def _rollout(self, config: Mapping[Sensor, Sequence[Action]], tracks: Set[Track], timestamp: datetime.datetime, end_time: datetime.datetime):
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

        if not len(self.rfis):
            return 0, np.inf

        # For each sensor in the configuration
        for sensor, actions in config.items():
            predicted_sensor = copy.deepcopy(sensor, memo)
            predicted_sensor.add_actions(actions)
            predicted_sensor.act(timestamp)
            if isinstance(sensor, Sensor):
                predicted_sensors.append(predicted_sensor)  # checks if its a sensor

        # Create dictionary of predictions for the tracks in the configuration
        predicted_tracks = set()
        for track in tracks:
            predicted_track = copy.copy(track)
            predicted_track.append(self.tracker._predictor.predict(predicted_track, timestamp=timestamp))
            time_interval = timestamp - predicted_track.timestamp
            prob_survive = np.exp(-self.tracker.prob_death* time_interval.total_seconds())
            track.exist_prob = prob_survive * track.exist_prob
            predicted_tracks.add(predicted_track)


        tracks_copy = [copy.copy(track) for track in tracks]

        for sensor in predicted_sensors:

            # Assumes one detection per track
            detections = {detection
                          for detection in sensor.measure(predicted_tracks, noise=False)
                          if isinstance(detection, TrueDetection)}

            center = (sensor.position[1], sensor.position[0])
            radius = sensor.fov_radius
            p = geodesic_point_buffer(*center, radius)
            self.tracker.prob_detect = _prob_detect_func([p])

            associations = self.tracker._associator.associate(tracks_copy, detections, timestamp)

            for track, multihypothesis in associations.items():
                if isinstance(self.tracker, SMCPHD_JIPDA):
                    # calculate each Track's state as a Gaussian Mixture of
                    # its possible associations with each detection, then
                    # reduce the Mixture to a single Gaussian State
                    posterior_states = []
                    posterior_state_weights = []
                    for hypothesis in multihypothesis:
                        posterior_state_weights.append(hypothesis.probability)
                        if hypothesis:
                            posterior_states.append(self.tracker._updater.update(hypothesis))
                        else:
                            posterior_states.append(hypothesis.prediction)

                    # Merge/Collapse to single Gaussian
                    means = StateVectors([state.state_vector for state in posterior_states])
                    covars = np.stack([state.covar for state in posterior_states], axis=2)
                    weights = np.asarray(posterior_state_weights)

                    post_mean, post_covar = gm_reduce_single(means, covars, weights)

                    track.append(GaussianStateUpdate(
                        np.array(post_mean), np.array(post_covar),
                        multihypothesis,
                        multihypothesis[0].prediction.timestamp))
                else:
                    if multihypothesis:
                        # Update track
                        state_post = self.tracker._updater.update(multihypothesis)
                        track.append(state_post)
                        track.exist_prob = Probability(1.)
                    else:
                        time_interval = timestamp - track.timestamp
                        track.append(multihypothesis.prediction)
                        non_exist_weight = 1 - track.exist_prob
                        prob_survive = np.exp(-self.tracker.prob_death * time_interval.total_seconds())
                        non_det_weight = prob_survive * track.exist_prob
                        track.exist_prob = non_det_weight / (non_exist_weight + non_det_weight)
        var = np.inf
        for rfi in self.rfis:
            xmin, ymin = rfi.region_of_interest.corners[0].longitude, rfi.region_of_interest.corners[0].latitude
            xmax, ymax = rfi.region_of_interest.corners[1].longitude, rfi.region_of_interest.corners[1].latitude
            geom = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])

            if rfi.task_type == TaskType.COUNT:
                target_types = [t.target_type.value for t in rfi.targets]
                _, var = calculate_num_targets_dist(tracks_copy, geom, target_types=target_types)
                if var < rfi.threshold_over_time.threshold[0]:
                    # TODO: Need to select the priority
                    config_metric += rfi.priority_over_time.priority[0]
                    if self.use_variance:
                        config_metric += 1/var
            elif rfi.task_type == TaskType.FOLLOW:
                for target in rfi.targets:
                    track = next((track for track in tracks_copy
                                  if track.id == str(target.target_UUID)), None)
                    if track is not None:
                        var = track.covar[0, 0] + track.covar[2, 2]
                        if var < rfi.threshold_over_time.threshold[0]:
                            config_metric += rfi.priority_over_time.priority[0]


        if timestamp == end_time:
            return config_metric, 0

        timestamp = timestamp + datetime.timedelta(seconds=1)

        all_action_choices = dict()
        for sensor in predicted_sensors:
            # get action 'generator(s)'
            action_generators = sensor.actions(timestamp)
            # list possible action combinations for the sensor
            action_choices = list(it.product(*action_generators))
            # dictionary of sensors: list(action combinations)
            all_action_choices[sensor] = action_choices

        # configs = list({sensor: action
        #                 for sensor, action in zip(all_action_choices.keys(), actionconfig)}
        #                for actionconfig in it.product(*all_action_choices.values()))
        configs = []
        poss = []
        for actionconfig in it.product(*all_action_choices.values()):
            cfg = dict()
            pos = set()
            for sensor, actions in zip(all_action_choices.keys(), actionconfig):
                action_x = next(
                    action for action in actions if action.generator.attribute == 'location_x')
                action_y = next(
                    action for action in actions if action.generator.attribute == 'location_y')
                cfg[sensor] = actions
                pos.add((action_x.target_value, action_y.target_value))
            if pos not in poss:
                configs.append(cfg)
                poss.append(pos)

        if len(configs) > self.num_samples:
            idx = np.random.choice(len(configs), self.num_samples, replace=False)
            configs = [configs[i] for i in idx]

        rewards = [self._rollout(config, tracks_copy, timestamp, end_time) for config in configs]
        config_metric += np.max(rewards)

        return config_metric, var


class RolloutPriorityRewardFunction2(RewardFunction):
    """A reward function which calculates the potential reduction in the uncertainty of track estimates
    if a particular action is taken by a sensor or group of sensors.

    Given a configuration of sensors and actions, a metric is calculated for the potential
    reduction in the uncertainty of the tracks that would occur if the sensing configuration
    were used to make an observation. A larger value indicates a greater reduction in
    uncertainty.
    """

    tracker: Tracker = Property(doc="Tracker used to track the tracks")
    timesteps: int = Property(doc="Number of timesteps to rollout")
    num_samples: int = Property(doc="Number of samples to take for each timestep", default=30)
    interval: datetime.timedelta = Property(doc="Interval between timesteps",
                                            default=datetime.timedelta(seconds=1))
    rfis: List[Any] = Property(doc="List of reward functions to use for prioritisation",
                               default=None)
    prob_survive: Probability = Property(doc="Probability of survival", default=Probability(0.99))
    use_variance: bool = Property(doc="Use variance in prioritisation", default=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.rfis is None:
            self.rfis = []

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

        if not len(self.rfis):
            return 0

        # Reward value
        end_time = metric_time + datetime.timedelta(seconds=self.timesteps)

        # Reward value
        config_metric, updated_tracks, predicted_sensors = self._compute_metric(config, tracks,
                                                                                metric_time)

        if metric_time == end_time:
            return config_metric

        timestamp = metric_time + datetime.timedelta(seconds=1)

        all_action_choices = dict()
        for sensor in predicted_sensors:
            # get action 'generator(s)'
            action_generators = sensor.actions(timestamp)
            # list possible action combinations for the sensor
            action_choices = list(it.product(*action_generators))
            # dictionary of sensors: list(action combinations)
            all_action_choices[sensor] = action_choices

        # configs = list({sensor: action
        #                 for sensor, action in zip(all_action_choices.keys(), actionconfig)}
        #                for actionconfig in it.product(*all_action_choices.values()))
        configs = []
        poss = []
        for actionconfig in it.product(*all_action_choices.values()):
            cfg = dict()
            pos = set()
            for sensor, actions in zip(all_action_choices.keys(), actionconfig):
                action_x = next(
                    action for action in actions if action.generator.attribute == 'location_x')
                action_y = next(
                    action for action in actions if action.generator.attribute == 'location_y')
                cfg[sensor] = actions
                pos.add((action_x.target_value, action_y.target_value))
            if pos not in poss:
                configs.append(cfg)
                poss.append(pos)

        if len(configs) > self.num_samples:
            idx = np.random.choice(len(configs), self.num_samples, replace=False)
            configs = [configs[i] for i in idx]

        rewards = [config_metric + self._rollout(config, updated_tracks, timestamp, end_time)
                   for config in configs]

        # Return value of configuration metric
        return np.max(rewards)

    def _compute_metric(self, config: Mapping[Sensor, Sequence[Action]], tracks: Set[Track],
                        timestamp: datetime.datetime):

        # Reward value
        config_metric = 0

        predicted_sensors = list()
        memo = {}

        # For each sensor in the configuration
        for sensor, actions in config.items():
            predicted_sensor = copy.deepcopy(sensor, memo)
            predicted_sensor.add_actions(actions)
            predicted_sensor.act(timestamp)
            if isinstance(sensor, Sensor):
                predicted_sensors.append(predicted_sensor)  # checks if its a sensor

        # Create dictionary of predictions for the tracks in the configuration
        predicted_tracks = set()
        for track in tracks:
            predicted_track = copy.copy(track)
            predicted_track.append(
                self.tracker._predictor.predict(predicted_track, timestamp=timestamp))
            time_interval = timestamp - predicted_track.timestamp
            prob_survive = np.exp(-self.tracker.prob_death * time_interval.total_seconds())
            track.exist_prob = prob_survive * track.exist_prob
            predicted_tracks.add(predicted_track)

        tracks_copy = [copy.copy(track) for track in tracks]

        for sensor in predicted_sensors:

            # Assumes one detection per track
            detections = {detection
                          for detection in sensor.measure(predicted_tracks, noise=False)
                          if isinstance(detection, TrueDetection)}

            center = (sensor.position[1], sensor.position[0])
            radius = sensor.fov_radius
            p = geodesic_point_buffer(*center, radius)
            self.tracker.prob_detect = _prob_detect_func([p])

            associations = self.tracker._associator.associate(tracks_copy, detections, timestamp)

            for track, multihypothesis in associations.items():
                if isinstance(self.tracker, SMCPHD_JIPDA):
                    # calculate each Track's state as a Gaussian Mixture of
                    # its possible associations with each detection, then
                    # reduce the Mixture to a single Gaussian State
                    posterior_states = []
                    posterior_state_weights = []
                    for hypothesis in multihypothesis:
                        posterior_state_weights.append(hypothesis.probability)
                        if hypothesis:
                            posterior_states.append(self.tracker._updater.update(hypothesis))
                        else:
                            posterior_states.append(hypothesis.prediction)

                    # Merge/Collapse to single Gaussian
                    means = StateVectors([state.state_vector for state in posterior_states])
                    covars = np.stack([state.covar for state in posterior_states], axis=2)
                    weights = np.asarray(posterior_state_weights)

                    post_mean, post_covar = gm_reduce_single(means, covars, weights)

                    track.append(GaussianStateUpdate(
                        np.array(post_mean), np.array(post_covar),
                        multihypothesis,
                        multihypothesis[0].prediction.timestamp))
                else:
                    if multihypothesis:
                        # Update track
                        state_post = self.tracker._updater.update(multihypothesis)
                        track.append(state_post)
                        track.exist_prob = Probability(1.)
                    else:
                        time_interval = timestamp - track.timestamp
                        track.append(multihypothesis.prediction)
                        non_exist_weight = 1 - track.exist_prob
                        prob_survive = np.exp(
                            -self.tracker.prob_death * time_interval.total_seconds())
                        non_det_weight = prob_survive * track.exist_prob
                        track.exist_prob = non_det_weight / (non_exist_weight + non_det_weight)
        var = np.inf
        for rfi in self.rfis:
            xmin, ymin = rfi.region_of_interest.corners[0].longitude, \
            rfi.region_of_interest.corners[0].latitude
            xmax, ymax = rfi.region_of_interest.corners[1].longitude, \
            rfi.region_of_interest.corners[1].latitude
            geom = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])

            if rfi.task_type == TaskType.COUNT:
                target_types = [t.target_type.value for t in rfi.targets]
                _, var = calculate_num_targets_dist(tracks_copy, geom, target_types=target_types)

                if var < rfi.threshold_over_time.threshold[0]:
                    # TODO: Need to select the priority
                    config_metric += rfi.priority_over_time.priority[0]
                    if self.use_variance:
                        config_metric += 1 / var
            elif rfi.task_type == TaskType.FOLLOW:
                for target in rfi.targets:
                    track = next((track for track in tracks_copy
                                  if track.id == str(target.target_UUID)), None)
                    if track is not None:
                        var = track.covar[0, 0] + track.covar[2, 2]
                        if var < rfi.threshold_over_time.threshold[0]:
                            config_metric += rfi.priority_over_time.priority[0]

        return config_metric, tracks_copy, predicted_sensors

    def _rollout(self, config: Mapping[Sensor, Sequence[Action]], tracks: Set[Track],
                 timestamp: datetime.datetime, end_time: datetime.datetime):
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

        if not len(self.rfis):
            return 0

        # Reward value
        config_metric, updated_tracks, predicted_sensors = self._compute_metric(config, tracks,
                                                                                timestamp)

        if timestamp == end_time:
            return config_metric

        timestamp = timestamp + datetime.timedelta(seconds=1)

        all_action_choices = dict()
        for sensor in predicted_sensors:
            # get action 'generator(s)'
            action_generators = sensor.actions(timestamp)
            # list possible action combinations for the sensor
            action_choices = list(it.product(*action_generators))
            # dictionary of sensors: list(action combinations)
            all_action_choices[sensor] = action_choices

        # configs = list({sensor: action
        #                 for sensor, action in zip(all_action_choices.keys(), actionconfig)}
        #                for actionconfig in it.product(*all_action_choices.values()))
        configs = []
        poss = []
        for actionconfig in it.product(*all_action_choices.values()):
            cfg = dict()
            pos = set()
            for sensor, actions in zip(all_action_choices.keys(), actionconfig):
                action_x = next(
                    action for action in actions if action.generator.attribute == 'location_x')
                action_y = next(
                    action for action in actions if action.generator.attribute == 'location_y')
                cfg[sensor] = actions
                pos.add((action_x.target_value, action_y.target_value))
            if pos not in poss:
                configs.append(cfg)
                poss.append(pos)

        idx = np.random.choice(len(configs), 1, replace=False)
        next_config = configs[idx[0]]

        config_metric += self._rollout(next_config, updated_tracks, timestamp, end_time)

        return config_metric


class RolloutPriorityRewardFunction3(RewardFunction):
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
    timesteps: int = Property(doc="Number of timesteps to rollout")
    num_samples: int = Property(doc="Number of samples to take for each timestep", default=30)
    interval: datetime.timedelta = Property(doc="Interval between timesteps",
                                            default=datetime.timedelta(seconds=1))
    rfis: List[Any] = Property(doc="List of reward functions to use for prioritisation",
                               default=None)
    prob_survive: Probability = Property(doc="Probability of survival", default=Probability(0.99))
    prob_death: Probability = Property(doc="Probability of death", default=Probability(0.01))
    use_variance: bool = Property(doc="Use variance in prioritisation", default=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.rfis is None:
            self.rfis = []

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
        end_time = metric_time + datetime.timedelta(seconds=self.timesteps)

        config_metric = self._rollout(config, tracks, metric_time, end_time)

        # Return value of configuration metric
        return config_metric

    def _rollout(self, config: Mapping[Sensor, Sequence[Action]], tracks: Set[Track], timestamp: datetime.datetime, end_time: datetime.datetime):
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

        if not len(self.rfis):
            return 0, np.inf

        # For each sensor in the configuration
        for sensor, actions in config.items():
            predicted_sensor = copy.deepcopy(sensor, memo)
            predicted_sensor.add_actions(actions)
            predicted_sensor.act(timestamp)
            if isinstance(sensor, Sensor):
                predicted_sensors.append(predicted_sensor)  # checks if its a sensor

        # Create dictionary of predictions for the tracks in the configuration
        predicted_tracks = set()
        for track in tracks:
            predicted_track = copy.copy(track)
            predicted_track.append(self.predictor.predict(predicted_track, timestamp=timestamp))
            time_interval = timestamp - predicted_track.timestamp
            prob_survive = np.exp(-self.prob_death* time_interval.total_seconds())
            track.exist_prob = prob_survive * track.exist_prob
            predicted_tracks.add(predicted_track)

        detected_tracks = set()
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

                # Replace prediction with update
                predicted_track.append(update)
                predicted_track.exist_prob = Probability(1.)
                detected_tracks.add(predicted_track)

        non_detected_tracks = predicted_tracks - detected_tracks
        for track in non_detected_tracks:
            time_interval = timestamp - track.timestamp
            non_exist_weight = 1 - track.exist_prob
            prob_survive = np.exp(-self.prob_death * time_interval.total_seconds())
            non_det_weight = prob_survive * track.exist_prob
            track.exist_prob = non_det_weight / (non_exist_weight + non_det_weight)

        var = np.inf
        for rfi in self.rfis:
            xmin, ymin = rfi.region_of_interest.corners[0].longitude, rfi.region_of_interest.corners[0].latitude
            xmax, ymax = rfi.region_of_interest.corners[1].longitude, rfi.region_of_interest.corners[1].latitude
            geom = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
            _, var = calculate_num_targets_dist(predicted_tracks, geom)
            if var < rfi.threshold:
                # TODO: Need to select the priority
                config_metric += rfi.priority_over_time.priority[0]
                if self.use_variance:
                    config_metric += 1/var


        if timestamp == end_time:
            return config_metric, 0

        timestamp = timestamp + datetime.timedelta(seconds=1)

        all_action_choices = dict()
        for sensor in predicted_sensors:
            # get action 'generator(s)'
            action_generators = sensor.actions(timestamp)
            # list possible action combinations for the sensor
            action_choices = list(it.product(*action_generators))
            # dictionary of sensors: list(action combinations)
            all_action_choices[sensor] = action_choices

        # configs = list({sensor: action
        #                 for sensor, action in zip(all_action_choices.keys(), actionconfig)}
        #                for actionconfig in it.product(*all_action_choices.values()))
        configs = []
        poss = []
        for actionconfig in it.product(*all_action_choices.values()):
            cfg = dict()
            pos = set()
            for sensor, actions in zip(all_action_choices.keys(), actionconfig):
                action_x = next(
                    action for action in actions if action.generator.attribute == 'location_x')
                action_y = next(
                    action for action in actions if action.generator.attribute == 'location_y')
                cfg[sensor] = actions
                pos.add((action_x.target_value, action_y.target_value))
            if pos not in poss:
                configs.append(cfg)
                poss.append(pos)

        if len(configs) > self.num_samples:
            idx = np.random.choice(len(configs), self.num_samples, replace=False)
            configs = [configs[i] for i in idx]

        rewards = [self._rollout(config, predicted_tracks, timestamp, end_time) for config in configs]
        config_metric += np.max(rewards)

        return config_metric, var


def _prob_detect_func(fovs):
    """Closure to return the probability of detection function for a given environment scan"""
    prob_detect = Probability(0.9)
    # Get the union of all field of views
    fovs_union = unary_union(fovs)
    if fovs_union.geom_type == 'MultiPolygon':
        fovs = [poly for poly in fovs_union]
    else:
        fovs = [fovs_union]

    paths = [Path(poly.boundary.coords) for poly in fovs]

    # Probability of detection nested function
    def prob_detect_func(state):
        for path_p in paths:
            if isinstance(state, ParticleState):
                prob_detect_arr = np.full((len(state),), Probability(0.1))
                points = state.state_vector[[0, 2], :].T
                inside_points = path_p.contains_points(points)
                prob_detect_arr[inside_points] = prob_detect
                return prob_detect_arr
            else:
                points = state.state_vector[[0, 2], :].T
                return prob_detect if np.alltrue(path_p.contains_points(points)) \
                    else Probability(0)

    return prob_detect_func