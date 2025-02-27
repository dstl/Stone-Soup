from abc import ABC
import copy
import datetime
from collections.abc import Mapping, Sequence
from typing import Set, Callable

from scipy.spatial import cKDTree

import numpy as np

from ..measures import KLDivergence
from ..platform import Platform
from ..sensormanager.action import Actionable
from ..types.detection import TrueDetection
from ..base import Base, Property
from ..predictor.base import Predictor
from ..predictor.particle import ParticlePredictor
from ..predictor.kalman import KalmanPredictor
from ..updater.kalman import ExtendedKalmanUpdater
from ..types.track import Track
from ..types.hypothesis import SingleHypothesis
from ..sensor.sensor import Sensor
from ..sensormanager.action import Action
from ..types.prediction import Prediction
from ..updater.base import Updater
from ..updater.particle import ParticleUpdater
from ..resampler.particle import SystematicResampler
from ..types.groundtruth import GroundTruthState
from ..dataassociator.base import DataAssociator
from ..types.state import StateVectors


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

    def __call__(self, config: Mapping[Sensor, Sequence[Action]], tracks: set[Track],
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
    return_tracks: bool = Property(default=False,
                                   doc="A flag for allowing the predicted track, "
                                       "used to calculate the reward, to be "
                                       "returned.")
    measurement_noise: bool = Property(default=False,
                                       doc="Decide whether or not to apply measurement model "
                                           "noise to the predicted measurements for sensor "
                                           "management.")

    def __call__(self, config: Mapping[Sensor, Sequence[Action]], tracks: set[Track],
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
        # For each sensor/platform in the configuration
        for actionable, actions in config.items():
            predicted_actionable = copy.deepcopy(actionable, memo)
            predicted_actionable.add_actions(actions)
            predicted_actionable.act(metric_time, noise=False)
            if isinstance(actionable, Sensor):
                predicted_sensors.add(predicted_actionable)  # checks if it's a sensor

        # Create dictionary of predictions for the tracks in the configuration
        predicted_tracks = set()
        for track in tracks:
            predicted_track = copy.copy(track)
            predicted_track.append(self.predictor.predict(predicted_track, timestamp=metric_time))
            predicted_tracks.add(predicted_track)

        for sensor in predicted_sensors:

            ground_truth_states = dict(
                (GroundTruthState(predicted_track.mean,
                                  timestamp=predicted_track.timestamp,
                                  metadata=predicted_track.metadata),
                 predicted_track)
                for predicted_track in predicted_tracks)

            detections_set = sensor.measure(
                set(ground_truth_states.keys()), noise=self.measurement_noise)

            # Assumes one detection per track
            detections = {
                ground_truth_states[detection.groundtruth_path]: detection
                for detection in detections_set
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
        if self.return_tracks:
            return config_metric, predicted_tracks
        else:
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

    predictor: Predictor = Property(default=None,
                                    doc="Predictor used to predict the track to a "
                                        "new state. This reward function is only "
                                        "compatible with :class:`~.ParticlePredictor` "
                                        "types.")
    updater: Updater = Property(default=None,
                                doc="Updater used to update the track to the new state. "
                                    "This reward function is only compatible with "
                                    ":class:`~.ParticleUpdater` types.")
    method_sum: bool = Property(default=True,
                                doc="Determines method of calculating reward."
                                    "Default calculates sum across all targets."
                                    "Otherwise calculates mean of all targets.")
    data_associator: DataAssociator = Property(default=None,
                                               doc="Data associator for associating "
                                                   "detections to tracks when "
                                                   "multiple sensors are managed.")

    return_tracks: bool = Property(default=False,
                                   doc="A flag for allowing the predicted track, "
                                       "used to calculate the reward, to be "
                                       "returned.")

    measurement_noise: bool = Property(default=False,
                                       doc="Decide whether or not to apply measurement model "
                                           "noise to the predicted measurements for sensor "
                                           "management.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.KLD = KLDivergence()

    def __call__(self, config: Mapping[Sensor, Sequence[Action]], tracks: set[Track],
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

        : Set[Track] (if defined)
            Set of tracks that have been predicted and updated in reward
            calculation if :attr:`return_tracks` is `True`

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
            predicted_track = copy.copy(track)
            if self.predictor:
                predicted_track.append(self.predictor.predict(track[-1],
                                                              timestamp=metric_time))
            else:
                predicted_track.append(Prediction.from_state(track[-1],
                                                             timestamp=metric_time))

            predicted_tracks.add(predicted_track)

        sensor_detections = self._generate_detections(predicted_tracks,
                                                      predicted_sensors,
                                                      timestamp=metric_time)
        det_count = 0
        for sensor, detections in sensor_detections.items():

            for predicted_track, detection_set in detections.items():
                det_count += len(detection_set)
                for n, detection in enumerate(detection_set):

                    # Generate hypothesis based on prediction/previous update and detection
                    hypothesis = SingleHypothesis(predicted_track.state, detection)

                    # Do the update based on this hypothesis and store covariance matrix
                    update = self.updater.update(hypothesis)

                    kld += self.KLD(predicted_track[-1], update)

                    if not isinstance(self, MultiUpdateExpectedKLDivergence):
                        predicted_track.append(update)

        if self.method_sum is False and det_count != 0:

            kld /= det_count

        # Return value of configuration metric
        if self.return_tracks:
            return kld, predicted_tracks
        else:
            return kld

    def _generate_detections(self, predicted_tracks, sensors, timestamp=None):

        all_detections = {}

        for sensor in sensors:
            detections = {}

            ground_truth_states = dict(
                (GroundTruthState(predicted_track.mean,
                                  timestamp=predicted_track.timestamp,
                                  metadata=predicted_track.metadata),
                 predicted_track)
                for predicted_track in predicted_tracks)

            detections_set = sensor.measure(
                set(ground_truth_states.keys()), noise=self.measurement_noise)

            # Assumes one detection per track
            detections = {
                ground_truth_states[detection.groundtruth_path]: {detection}
                for detection in detections_set
                if isinstance(detection, TrueDetection)}

            if self.data_associator:
                tmp_hypotheses = self.data_associator.associate(
                    predicted_tracks,
                    {det for dets in detections.values() for det in dets},
                    timestamp)
                detections = {predicted_track: {hypothesis.measurement}
                              for predicted_track, hypothesis in tmp_hypotheses.items()
                              if hypothesis}

            all_detections.update({sensor: detections})

        return all_detections


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

    predictor: ParticlePredictor = Property(default=None,
                                            doc="Predictor used to predict the track to a "
                                                "new state. This reward function is only "
                                                "compatible with :class:`~.ParticlePredictor` "
                                                "types.")
    updater: ParticleUpdater = Property(default=None,
                                        doc="Updater used to update the track to the new state. "
                                            "This reward function is only compatible with "
                                            ":class:`~.ParticleUpdater` types.")

    updates_per_track: int = Property(default=2,
                                      doc="Number of measurements to generate from each "
                                          "track prediction. This should be > 1.")

    measurement_noise: bool = Property(default=True)

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

    def _generate_detections(self, predicted_tracks, sensors, timestamp=None):

        detections = {}
        all_detections = {}
        resampler = SystematicResampler()

        for sensor in sensors:
            for predicted_track in predicted_tracks:

                measurement_sources = resampler.resample(predicted_track[-1],
                                                         nparts=self.updates_per_track)
                tmp_detections = set()
                for state in measurement_sources.state_vector:
                    tmp_detections.update(
                        sensor.measure({GroundTruthState(state,
                                                         timestamp=timestamp,
                                                         metadata=predicted_track.metadata)},
                                       noise=self.measurement_noise))

                detections.update({predicted_track: tmp_detections})
            all_detections.update({sensor: detections})

        return all_detections


class InformationCoverageReward(RewardFunction):
    """Reward function that encourages coverage of a distribution based on the
    current state of information about the environment and a reference level
    of information. Information in this context is represented by a set
    points in the environment which each have an information value (information state).
    The information state is decayed through time, with :attr:`decay_rate`,
    and new information from sensing actions, calculated by :attr:`sensing_info_func`,
    added to the information state. This reward compares the difference between a
    user defined information setpoint and the current information setpoint and
    rewards minimisation of the difference between the two, focussed on an
    underlying density. In the absence of target tracks, this is a uniform
    density across the environment and in the presence of a target, this density
    is a mixture between a uniform density and the target estimate density, with
    :attr:`search_weight` defining the mixture composition. """

    predictor: ParticlePredictor = Property(
        doc="Predictor used to predict the track to a new state. This reward "
        "function is only compatible with :class:`~.ParticlePredictor` types."
    )

    updater: ParticleUpdater = Property(
        doc="Updater used to update the track to the new state. This reward "
        "function is only compatible with :class:`~.ParticleUpdater` types."
    )

    environment_cells: StateVectors = Property(
        doc="Coordinates defining cells approximating the environment for "
        "the information state. Vector of shape (2, n) where n is the number of cells."
    )

    reference_information: StateVectors = Property(
        doc="Reference level of information to achieve in each grid cell. "
        "Vector of shape (1, n) where n is the number of cells. This should "
        "match :attr:`environment_cells`."
    )

    sensing_info_func: Callable = Property(
        doc="Function describing the information from a sensing action."
    )

    track_thresh_func: Callable = Property(
        doc="Callable function that implements the trigger for including track "
        "information in the reward function. Allows the user to take advantage "
        "of specific filter properties, such as the existence probability in "
        "the Bernoulli Filter. Takes input of a track object and must return "
        "`True` or `False`."
    )

    information_decay: float = Property(
        default=-0.05,
        doc="The decay rate of information acquired by sensing actions. Higher "
        "value leads to historical actions being *forgotten* in less time. Value "
        "should be 0 or negative."
    )

    measurement_noise: bool = Property(
        default=False,
        doc="Decide whether or not to apply measurement model noise to the "
        "predicted measurements for sensor management."
    )

    return_tracks: bool = Property(
        default=False,
        doc="A flag for allowing the predicted track, used to calculate the "
        "reward, to be returned."
    )

    position_mapping: Sequence[int] = Property(
        default=(0, 1),
        doc="Mapping for the :math:`x`, :math:`y` position of the target state vector."
    )

    search_weight: float = Property(
        default=0.5,
        doc="Density used in the reward function is a mixture between target "
        "density and search density. This controls the weighting of the mixture."
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._environment_cell_tree = cKDTree(self.environment_cells.T)

    def __call__(self, config: Mapping[Sensor, Sequence[Action]], tracks: Set[Track],
                 metric_time: datetime.datetime, *args, **kwargs):
        """
        For a given configuration of sensors and actions this reward function
        calculates the information coverage reward based on sensing actions and the
        current information state stored in track metadata. It is calculated by
        subtracting the updated information state from the information reference,
        multiplied into the coverage density.

        This requires a mapping of sensors to action(s) to be evaluated by the
        reward function, a set of tracks at given time and the time at which
        the actions would be carried out until. Track metadata should contain
        the information state, which gets updated outside of sensor management
        between calls to :meth:`choose_action`, reflecting the impact of the
        selected action. This has currently been tested with tracks containing
        a single :class:`~.Track`.

        The metric returned is the information coverage reward calculated
        based on all tracks.

        Returns
        -------
        : float
            Information coverage reward for given configuration

        : Set[Track] (if defined)
            Set of tracks that have been predicted and updated in reward
            calculation if :attr:`return_tracks` is `True`

        """
        # Reward value
        config_metric = 0

        predicted_sensors = set()
        memo = {}
        # For each sensor/platform in the configuration
        for actionable, actions in config.items():
            predicted_actionable = copy.deepcopy(actionable, memo)
            predicted_actionable.add_actions(actions)
            predicted_actionable.act(metric_time, noise=False)
            if isinstance(actionable, Sensor):
                predicted_sensors.add(predicted_actionable)  # checks if it's a sensor

        # If no actionables provided then return.
        if len(config.items()) > 0:
            # Calculate sample time for information decal. Assumes all actionables have the same
            sample_time = metric_time - actionable.timestamp

            # calculate the density used in the reward. Initialise to a uniform value
            overall_density = 1/self.environment_cells.shape[1] \
                * np.ones((1, self.environment_cells.shape[1]))
            n_tracks = len(tracks)

            # Create set of predictions for the tracks in the configuration
            predicted_tracks = set()
            for track in tracks:
                predicted_track = copy.copy(track)
                information_state = copy.copy(predicted_track.metadata['information_state'])

                # Predict track if it passes the track_thresh_func
                if self.track_thresh_func(predicted_track):
                    predicted_track.append(Prediction.from_state(track[-1],
                                                                 timestamp=metric_time))
                else:
                    predicted_track.append(self.predictor.predict(predicted_track,
                                                                  timestamp=metric_time))
                predicted_tracks.add(predicted_track)

            for sensor in predicted_sensors:

                # Assumes one detection per track
                detections = {predicted_track: detection
                              for detection in
                              sensor.measure({GroundTruthState(predicted_track.mean,
                                                               timestamp=predicted_track.timestamp,
                                                               metadata=predicted_track.metadata)},
                                             noise=self.measurement_noise)
                              for predicted_track in predicted_tracks
                              if isinstance(detection, TrueDetection)}

                # Only update is track passes track_thresh_func
                for predicted_track, detection in detections.items():
                    if self.track_thresh_func(predicted_track):
                        # Generate hypothesis based on prediction/previous update and detection
                        hypothesis = SingleHypothesis(predicted_track.state, detection)

                        # Do the update based on this hypothesis and store covariance matrix
                        update = self.updater.update(hypothesis)

                        # Replace prediction with update
                        predicted_track.append(update)

            # Construct the coverage density
            for predicted_track in predicted_tracks:
                if self.track_thresh_func(predicted_track):
                    # Calculate which environment_cell points that predicted track
                    # particle states are closest to
                    _, nearest_cells = self._environment_cell_tree.query(
                        predicted_track.state.state_vector[self.position_mapping,
                                                           :].T)

                    # Multiply currently uniform density by search weight
                    overall_density *= self.search_weight

                    # Get weight values and multiply them by (1-search weight) and (1/n_tracks)
                    weight_vals = \
                        np.exp(np.log((1-self.search_weight)*(1/n_tracks))
                               + predicted_track.log_weight)

                    # Add weight values to each environment cell point that
                    # particles are closest to
                    np.add.at(overall_density, (0, nearest_cells), weight_vals)

            # update information state according to candidate sensing and platform actions
            information_state = self.update_information_state(predicted_sensors,
                                                              information_state,
                                                              sample_time)

            # calculate error between reference information and information state
            information_error = self.reference_information - information_state
            # Ensure no negative values remain in the error to prevent
            # penalisation of gaining too much information.
            information_error[information_error < 0] = 0.

            # Calculate reward
            config_metric = -1*np.sum(information_error * overall_density)

        # Return value of configuration metric
        if self.return_tracks:
            # if returning tracks, update track information state in metadata
            for track in predicted_tracks:
                track.metadata['information_state'] = information_state
            return config_metric, predicted_tracks
        else:
            return config_metric

    def update_information_state(self, sensors, information_state, sample_time):
        # Implements the information model, which decays the current information
        # state and adds latest information.

        # call user defined sensing information function to calculate information
        # from sensing actions
        sensing_info = self.sensing_info_func(sensors, self.environment_cells)

        # Calculate information rate based on decay on new sensing actions
        information_rate = information_state*self.information_decay + sensing_info

        # Calculate updated information state
        information_state += information_rate*sample_time.total_seconds()

        return information_state
