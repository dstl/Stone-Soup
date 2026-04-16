from abc import ABC
import copy
import datetime
from collections.abc import Mapping, Sequence

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


class AdditiveRewardFunction(RewardFunction):
    """Additive reward function

    Elementwise addition of corresponding reward functions.
    """

    reward_function_list: Sequence[RewardFunction] = Property(doc="List of reward functions")
    weights: list = Property(default=None, doc="Weight for each reward function.")

    def __call__(self, config: Mapping[Sensor, Sequence[Action]], tracks: set[Track],
                 metric_time: datetime.datetime, *args, **kwargs):
        if self.weights is None:
            self.weights = [1] * len(self.reward_function_list)
        if len(self.reward_function_list) != len(self.weights):
            raise IndexError
        return np.sum([reward_function(config, tracks, metric_time, *args, **kwargs) * weight
                       for reward_function, weight in
                       zip(self.reward_function_list, self.weights)])


class MultiplicativeRewardFunction(RewardFunction):
    """Multiplicative reward function

    Elementwise multiplication of corresponding reward functions.
    """

    reward_function_list: Sequence[RewardFunction] = Property(doc="List of reward functions")
    weights: list = Property(default=None, doc="Weight for each reward function.")

    def __call__(self, config: Mapping[Sensor, Sequence[Action]], tracks: set[Track],
                 metric_time: datetime.datetime, *args, **kwargs):
        if self.weights is None:
            self.weights = [1] * len(self.reward_function_list)
        if len(self.reward_function_list) != len(self.weights):
            raise IndexError
        return np.prod([reward_function(config, tracks, metric_time, *args, **kwargs) * weight
                       for reward_function, weight in
                       zip(self.reward_function_list, self.weights)])


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


class QuadraticInformationGain(RewardFunction, QuadraticDistance):
    """
    The quadratic information gain reward function. An implementation 
    is provided for the GM-PHD filter under the Gaussian kernel parametrisation.
    """

    num_samples: int = Property(doc='Number of samples to use in the Monte Carlo computation of the measurement average')
    
    filter_data: dict = Property(doc='Dictionary containing data for the particular filter model in question')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_dim = self.filter_data['state dimension']
        #########################################
        ########### Gaussian kernel #############
        #########################################
        if self.kernel == 'Gaussian':

            # unpack kernel specific parameters
            allowed_keys = {'covariance'}

            if self.kernel_parameters is not None:
                unknown = set(self.kernel_parameters) - allowed_keys
                if unknown:
                    raise ValueError(f"Unknown parameter(s) for vectorised_gaussian_eval: {', '.join(unknown)}.")
                
                R = self.kernel_parameters['covariance']

                # check dimension symmetry and positive-definiteness
                if R.shape != (self.state_dim, self.state_dim) or not np.allclose(R, R.T, rtol=1e-10, atol=1e-10) or np.any(np.linalg.eigvals(R) < 0):
                    raise ValueError(f'The {self.kernel} kernel covariance matrix must be symmetric and positive-definite with shape ({self.state_dim}, {self.state_dim}).')
            else:
                raise ValueError(f'No covariance matrix was provided for the {self.kernel} kernel.')
        else: 
            raise NotImplementedError(f'The Quadratic Information Gain with the {self.kernel} kernel parametrisation is not implemented.')
            
        #################################
        ######### GM-PHD Update #########
        #################################
        if self.filter_data['filter model'] != 'GMPHD':
            raise NotImplementedError(f'The Quadratic Information Gain for the {self.filter_data['filter model']} filter is not implemented.')
        

    def __call__(self, config: Mapping[Sensor, Sequence[Action]], tracks: Set[Track],
                 metric_time: datetime.datetime, *args, **kwargs):
        
        reward = 0

        # compute gm-phd prediction gaussian mixture given the previous gaussian mixture
        pred_wghts=[]
        pred_means=[]
        pred_covcs=[]
        for n, track in enumerate(tracks):
            # print('prediction')
            propagated_track = self.filter_data['predictor'].predict(track, timestamp=track.timestamp + datetime.timedelta(seconds=1))
            pred_wghts.append(track.weight * self.filter_data['survival probability'])
            pred_means.append(np.array(propagated_track.mean.flatten()))
            pred_covcs.append(np.array(propagated_track.covar))
        pred_wghts = np.asarray(pred_wghts)
        pred_means = np.asarray(pred_means)
        pred_covcs = np.asarray(pred_covcs)

        predicted_mixture = [pred_wghts, pred_means, pred_covcs]

        if len(predicted_mixture[0]) == 0:
            reward = np.random.rand()
            return reward # return a random reward if there are no predicted states, this leads to random action selection

        # extract all sensors in this configuration
        memo = {}
        predicted_sensors = set()
        # For each actionable in the configuration
        for actionable, actions in config.items():
            # print('unpacking sensors')
            # Don't currently have an Actionable base for platforms hence either Platform or Sensor
            if isinstance(actionable, Platform) or isinstance(actionable, Actionable):
                predicted_actionable = copy.deepcopy(actionable, memo)
                predicted_actionable.add_actions(actions)
                predicted_actionable.act(metric_time)
                if isinstance(actionable, Sensor):
                    predicted_sensors.add(predicted_actionable)  # checks if its a sensor
                elif isinstance(actionable, Platform):
                    predicted_sensors.update(predicted_actionable.sensors)


        for sensor in predicted_sensors:
            # print('eval for each sensor')
            ### sampling ###
            #setup categorical distribution
            num_samples = self.num_samples
            num_choices = len(predicted_mixture[0]) + 1 # +1 for the clutter distribution
            weights_list = [self.filter_data['clutter rate'], *list(self.filter_data['detection probability']*np.array(predicted_mixture[0]))]
            
            normalised_weights = weights_list/np.sum(weights_list)
            choices = np.random.choice(np.arange(num_choices), p=normalised_weights, size=num_samples)
            
            #generate samples from the measurement distribution. ONLY WORKING FOR POLAR SENSORS
            p_sampled = False # flag for if the prediction gets sampled
            samples = np.zeros((num_samples, 2))
            for n in range(num_choices):
                # print('sampling')
                idx = np.where(choices == n)[0]
                if idx.size:
                    if n == 0:
                        #clutter
                        thetas = np.random.uniform(sensor.dwell_centre.item()-sensor.fov_angle/2, sensor.dwell_centre.item()+sensor.fov_angle/2,  size=idx.size)
                        rs = np.random.uniform(0, sensor.max_range, size=idx.size)
                        samps = np.stack((thetas, rs), axis=1)
                        
                    else:
                        #prediction
                        p_sampled = True
                        jacobian = sensor.measurement_model.jacobian(GroundTruthState(state_vector=predicted_mixture[1][n-1]))
                        samps = np.random.multivariate_normal(mean = np.array([float(x) for x in np.array(cart2pol(predicted_mixture[1][n-1][0]-sensor.position.flatten()[0], predicted_mixture[1][n-1][2]-sensor.position.flatten()[1]))[::-1]]),
                                                            cov = sensor.noise_covar + jacobian @ predicted_mixture[2][n-1] @ jacobian.T, size=idx.size)
                        
                        #convert theta to [-pi, pi] and range to [0, +inf)
                        samps[:,0] = (samps[:,0] + np.pi) % (2*np.pi) - np.pi
                        samps[:,1] = np.abs(samps[:,1])

                        #check which samples are in fov
                        remove_mask = []
                        for id_n, id in enumerate(idx):
                            if not (sensor.dwell_centre.item()-sensor.fov_angle/2 <= samps[id_n][0] <= sensor.dwell_centre.item()+sensor.fov_angle/2) or not (samps[id_n][1] <= sensor.max_range):
                                remove_mask.append(id)

                    samples[idx] = samps
            
            # remove out of fov samples
            if p_sampled:
                samples = np.delete(samples, remove_mask, axis=0)
                num_samples = len(samples)

            # return a reward of 0 if number of samples is zero
            if num_samples == 0:
                return reward
            
            #project prediction components according to the jacobian
            jacobians = [sensor.measurement_model.jacobian(GroundTruthState(state_vector=predicted_mixture[1][k])) for k in range(len(predicted_mixture[1]))]
            jacobians_T = [jaco.T for jaco in jacobians]
            projected_prediction_means = np.asarray([jacobians[n] @ predicted_mixture[1][n].T for n in range(len(predicted_mixture[1]))])
            projected_predicted_covs_half = np.asarray([predicted_mixture[2][n] @ jacobians_T[n] for n in range(len(predicted_mixture[1]))])
            half_projected_predicted_covs = np.asarray([jacobians[n] @ predicted_mixture[2][n] for n in range(len(predicted_mixture[1]))])
            projected_predicted_covs = np.asarray([jacobians[n] @ predicted_mixture[2][n] @ jacobians_T[n] for n in range(len(predicted_mixture[1]))])

            #compute reward
            qs = self.vectorised_gaussian_eval(False, True, dim=2, w1=predicted_mixture[0], m2=samples, m1=projected_prediction_means, const_cov=sensor.noise_covar, var_cov1= projected_predicted_covs).flatten() # j, z
            Ks = projected_predicted_covs_half @ np.linalg.inv(sensor.noise_covar + projected_predicted_covs) # i
            means = (predicted_mixture[1][:, None, :] + (Ks[:, None, :, :] @ (samples[None, :, :] - projected_prediction_means[:, None, :])[..., None])[..., 0]).reshape(-1, 4) # i, z
            covs = np.repeat(predicted_mixture[2]-Ks @ half_projected_predicted_covs, num_samples, axis=0) # i
            gamma = self.vectorised_gaussian_eval(False, True, dim=4, m1=means, m2=means, const_cov=self.kernel_parameters['covariance'], var_cov1=covs, var_cov2= covs) # i, j, z
            rhos2 = (self.filter_data['clutter rate'] + self.filter_data['detection probability']*np.sum([predicted_mixture[0][k]*qs.reshape(len(predicted_mixture[0]), num_samples)[k] for k in range(len(predicted_mixture[0]))]))**2 # z

            rhos2 = np.maximum(rhos2, 1e-12) # prevent divide by zero errors


            reward += (self.filter_data['detection probability']**2/num_samples) * np.sum(np.repeat(predicted_mixture[0], num_samples)[None, :] * qs[None, :] * gamma / (rhos2))
            
            reward *= self.filter_data['clutter rate'] + self.filter_data['detection probability'] * np.sum(predicted_mixture[0]) # Monte Carlo estimator normalisation constant
            
        return reward
        
