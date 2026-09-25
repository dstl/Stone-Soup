import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.special import logsumexp
from stonesoup.base import Property
from stonesoup.tracker.base import Tracker, _TrackerMixInNext
from stonesoup.types.track import Track
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.particle import Particle
from stonesoup.types.numeric import Probability
from stonesoup.types.state import ParticleState
from stonesoup.types.prediction import Prediction


class _BaseFixedLagTracker(_TrackerMixInNext, Tracker):

    initiator = Property('Initiator used to initialise the track')
    deleter = Property('Deleter used to delete the track')
    detector = Property('Detector to generate detections')
#    data_associator = Property('Data associator to associate detections to tracks')
    predictor = Property('Predictor to predict the next state')
    updater = Property('Updater to update the tracks with the measurements')
    lag = Property('The lag in time steps',
                   default=3)
    _lag_assoc_buffer = Property('Buffer to store the associations for the lag',
                                 default=[])

    # initialisation
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._track = None

    def __iter__(self):
        self.detector_iter = iter(self.detector)
        return super().__iter__()

    @property
    def tracks(self) -> set[Track]:
        return {self._track} if self._track else set()

    def _get_detections(self):
        time, detections = next(self.detector_iter)

        # validate the number of detections
        timestamps = {detection.timestamp for detection in detections}

        if len(timestamps) > 1:
            raise ValueError("All detections must have the same timestamp")

        for temp_time in timestamps:
            if temp_time != time:
                time = temp_time
                break
        return time, detections

    def _fill_lag_buffer(self, associations):
        """
            Fill the lag storage with associations
        """
        self._lag_assoc_buffer.append(associations)

        if associations['detections']:
            for det in associations['detections']:
                hyphotesis = SingleHypothesis(prediction=associations['prediction'],
                                              measurement=det)
                # assuming single measurement for simplicity
            state_post = self.updater.update(hyphotesis)
            # assuming single measurement for simplicity
            self._track.append(state_post)
        else:
            self._track.append(associations['prediction'])

    def _process_lag(self):
        raise NotImplementedError


class FixedLagTracker(_BaseFixedLagTracker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._track = None  # current track

    @property
    def tracks(self) -> set[Track]:
        return {self._track} if self._track else set()

    def _process_lag(self):
        """
        Main function to process the lag, update the states and weights in the
        lag buffer and then update the track with the new measurement
        """

        new_weights_lag = {}  # use the dictionary in case to maximise the transfer on the MTT case
        keys = ('logprior', 'logposterior', 'oldweights')
        default_value = np.zeros_like(self._track[-1].log_weight)
        struct = dict.fromkeys(keys, default_value)

        # reorder the lag buffer
        self._lag_assoc_buffer.sort(key=lambda x: x['time'])

        # loop in the lag buffer
        for i_lag, element in enumerate(self._lag_assoc_buffer):

            _prior = element['prior']

            lag_position = len(self._track) - (self.lag + 1 - i_lag)

            _old_weights = _prior.log_weight

            if element['detections']:
                _meas = [det for det in element['detections']][0]
            else:
                _meas = None

            time_diff = element['time'] - _prior.timestamp

            new_prediction = self.predictor.predict(_prior,
                                                    _meas.timestamp if _meas else element['time'])

            logprior = self.predictor.transition_model.logpdf(new_prediction,
                                                              _prior,
                                                              time_interval=time_diff)

            if self.updater.measurement_model is None and _meas is not None:
                detection_meas_model = _meas.measurement_model
                logposterior = detection_meas_model.logpdf(_meas, new_prediction)
            elif self.updater.measurement_model is not None and _meas is not None:
                logposterior = self.updater.measurement_model.logpdf(_meas, new_prediction)
            else:
                logposterior = 0  # case of no measurements

            # update the values over the new lag values
            struct['logprior'] += logprior
            struct['logposterior'] += logposterior
            struct['oldweights'] += _old_weights

            # save them over the lag
            new_weights_lag[self._track] = struct

            temp_weights = _old_weights + logposterior - logprior
            temp_weights -= logsumexp(temp_weights)

            self._track[lag_position] = new_prediction  # update the state in the lag position
            self._track[lag_position].log_weight = temp_weights  # update the weights

        new_weights = (new_weights_lag[self._track]['oldweights'] +
                       new_weights_lag[self._track]['logposterior'] -
                       new_weights_lag[self._track]['logprior'])

        # normalise the weights
        new_weights -= logsumexp(new_weights)

        track_pos = len(self._track) - (self.lag + 1)
        # now update the weights at the lag position
        self._track[track_pos].log_weight = new_weights

        # in case do the resample
        self._track[track_pos] = self.updater.resampler.resample(self._track[track_pos])

        # after considered we should ditch the oldest element
        self._lag_assoc_buffer.pop(0)

    def _identify_lag_position(self, current_time):
        """
        Function to identify the correct track position and prior
        """

        for idx, state in enumerate(self._track):
            if state.timestamp == current_time:
                if idx == 0:
                    return idx
                else:
                    return idx-1
            else:
                continue
        return -1

    # main tracking loop
    def __next__(self):
        # obtain the detections
        time, detections = self._get_detections()

        if self._track is not None:

            lag_index = self._identify_lag_position(time)
            _current_prior = self._track[lag_index]

            _current_prediction = self.predictor.predict(_current_prior,
                                                         time)

            current_data = {'prior': _current_prior,
                            'prediction': _current_prediction,
                            'detections': detections,
                            'time': time}

            # lets catch the case of lag 0 ~ standard pf
            if self.lag == 0:
                if current_data['detections']:
                    for det in current_data['detections']:
                        hyphotesis = SingleHypothesis(prediction=current_data['prediction'],
                                                      measurement=det)
                        # assuming single measurement for simplicity
                    state_post = self.updater.update(hyphotesis)
                    # assuming single detection for simplicity
                    self._track.append(state_post)
                else:
                    self._track.append(current_data['prediction'])

            # case of no lag of before lag
            elif len(self._lag_assoc_buffer) < self.lag+1:
                self._fill_lag_buffer(current_data)

            else:
                self._process_lag()

                self._lag_assoc_buffer.append(current_data)
                # order it
                self._lag_assoc_buffer.sort(key=lambda x: x['time'])

                # update the track with the new measurement
                if self._lag_assoc_buffer[-1]['detections']:
                    for det in self._lag_assoc_buffer[-1]['detections']:
                        current_pred = self._lag_assoc_buffer[-1]['prediction']
                        hyphotesis = SingleHypothesis(prediction=current_pred,
                                                      measurement=det)
                        # assuming single measurement for simplicity
                    state_post = self.updater.update(hyphotesis)
                    # assuming single detection for simplicity
                    self._track.append(state_post)
                else:
                    self._track.append(self._lag_assoc_buffer[-1]['prediction'])

        if self._track is None or self.deleter.delete_tracks(self.tracks):
            new_tracks = self.initiator.initiate(detections, time)
            if new_tracks:
                self._track = new_tracks.pop()
            else:
                self._track = None

        return time, self.tracks


class FixedLagNutsTracker(_BaseFixedLagTracker):
    """
    Implementation of the Fixed Lag Tracker using the No-U-Turn Sampler
    (NUTS) to process the lag and update the track with a sampling on the trajectory.

    This implementation follows the papers:
        [1] Varsi, A., Devlin, L., Horridge, P., & Maskell, S. (2024).
        A general-purpose fixed-lag no-u-turn sampler for nonlinear non-gaussian
        state space models. IEEE Transactions on Aerospace and Electronic Systems.
    """

    sampler = Property('NUTS sampler to use in the fixed lag tracker')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._track = None  # current track
        if self.lag != self.sampler.lag_size:
            raise ValueError("Lag size in tracker and sampler must be the same")

    @property
    def tracks(self) -> set[Track]:
        return {self._track} if self._track else set()

    def _identify_lag_position(self, current_time):
        """
        Function to identify the correct track position and prior
        """

        for idx, state in enumerate(self._track):
            if state.timestamp == current_time:
                if idx == 0:
                    return idx
                else:
                    return idx-1
            else:
                continue
        return -1

    def _process_lag_nuts(self):
        """
        Custom function to process the lag using the NUTS sampler
        """

        # we might need to do the same to adjust how it works
        new_weights_lag = {}
        keys = ('logprior', 'logposterior', 'lag_weights', 'lag_k_1')
        default_value = np.zeros_like(self._track[-1].log_weight)
        struct = dict.fromkeys(keys, default_value)

        self._lag_assoc_buffer.sort(key=lambda x: x['time'])  # order the lag buffer

        narrow_down_to_tracks = self._lag_assoc_buffer

        # relevant nuts samples
        new_x, new_v, old_v = self.sampler.rvs(narrow_down_to_tracks)

        # loop in the lag buffer
        lkernel_logpdf = mvn.logpdf(-new_v.T, mean=np.zeros(self.sampler.MM.shape[1]),
                                    cov=self.sampler.MM[0, :, :])  # this is just n_particle size
        q_logpdf = mvn.logpdf(old_v.T, mean=np.zeros(self.sampler.MM.shape[1]),
                              cov=self.sampler.MM[0, :, :])

        ghost_logprior = self.predictor.transition_model.logpdf(
            narrow_down_to_tracks[-1]['prediction'],
            narrow_down_to_tracks[-1]['prior'],
            time_interval=(narrow_down_to_tracks[-1]['prediction'].timestamp -
                           narrow_down_to_tracks[-1]['prior'].timestamp))

        for i_lag, element in enumerate(narrow_down_to_tracks):
            lag_position = len(self._track) - (self.lag + 1 - i_lag)
            if i_lag == 0:  # use the associations prior
                _prior = element['prior']
                self.num_dims = _prior.state_vector.shape[0]
                self.num_particles = _prior.state_vector.shape[1]
                struct['lag_k_1'] += _prior.log_weight + ghost_logprior  # prev k-1
            else:
                # use the just updated track position
                _prior = element['prior']  # this should be the prior at the lag position

            _old_weights = _prior.log_weight

            if element['detections']:
                _meas = [det for det in element['detections']][0]
                # this is just to consider the single measurement case,
                # need to be changed for multi measurement
            else:
                _meas = None

            time_diff = element['time'] - _prior.timestamp  # maybe we have it somewhere else

            particles = new_x[i_lag*self.num_dims: (i_lag+1)*self.num_dims, :]
            particle_list = [
                Particle(state_vector=particles[:, i].reshape(-1, 1),
                         weight=Probability(1. / self.num_particles)
                         ) for i in range(self.num_particles)]

            temp_state = ParticleState(state_vector=None,
                                       particle_list=particle_list,
                                       timestamp=element['time'])

            new_prediction = Prediction.from_state(
                temp_state,
                parent=_prior,
                state_vector=temp_state.state_vector,
                timestamp=temp_state.timestamp,
                transition_model=self.predictor.transition_model,
                prior=_prior)

            logprior = self.predictor.transition_model.logpdf(new_prediction,
                                                              _prior,
                                                              time_interval=time_diff)

            if self.updater.measurement_model is None and _meas is not None:
                detection_meas_model = _meas.measurement_model
                logposterior = detection_meas_model.logpdf(_meas, new_prediction)
            elif self.updater.measurement_model is not None and _meas is not None:
                logposterior = self.updater.measurement_model.logpdf(_meas, new_prediction)
            else:
                logposterior = 0

            # update the values over the new lag values
            struct['logprior'] += logprior  # this is not needed
            struct['logposterior'] += logposterior
            struct['lag_weights'] += _old_weights

            # save them over the lag
            new_weights_lag[self._track] = struct

            temp_weights = _old_weights + logposterior
            temp_weights -= logsumexp(temp_weights)
            self._track[lag_position] = new_prediction  # update the state in the lag position
            self._track[lag_position].log_weight = temp_weights  # update the weights

        new_weights = (new_weights_lag[self._track]['lag_weights'] +
                       new_weights_lag[self._track]['logposterior'] -
                       new_weights_lag[self._track]['lag_k_1'] +
                       lkernel_logpdf -
                       q_logpdf -
                       new_weights_lag[self._track]['logprior'])

        # normalise the weights
        new_weights -= logsumexp(new_weights)

        lag_position = len(self._track) - (self.lag + 1)
        # now update the weights at the lag position
        self._track[lag_position].log_weight = new_weights

        # in case do the resample
        self._track[lag_position] = self.updater.resampler.resample(self._track[lag_position])

        # after considered we should ditch the oldest element
        self._lag_assoc_buffer.pop(0)

    # main tracking loop
    def __next__(self):

        # ok obtain the detections
        time, detections = self._get_detections()

        if self._track is not None:

            # predict
            lag_index = self._identify_lag_position(time)
            _current_prior = self._track[lag_index]
            _current_prediction = self.predictor.predict(_current_prior,
                                                         time)

            current_data = {'prior': _current_prior,
                            'prediction': _current_prediction,
                            'detections': detections,
                            'time': time}

            # case of no lag of before lag
            if len(self._lag_assoc_buffer) < self.lag+1:
                # fill the lag buffer with all the data needed
                self._fill_lag_buffer(current_data)

            else:
                # here begins the fix lag component to understand how to integrate
                self._process_lag_nuts()

                self._lag_assoc_buffer.append(current_data)

                # order it
                self._lag_assoc_buffer.sort(key=lambda x: x['time'])

                # update the track with the new measurement
                if self._lag_assoc_buffer[-1]['detections']:
                    for det in self._lag_assoc_buffer[-1]['detections']:
                        _prediction = self._lag_assoc_buffer[-1]['prediction']
                        hyphotesis = SingleHypothesis(prediction=_prediction,
                                                      measurement=det)
                        # assuming single measurement for simplicity
                    state_post = self.updater.update(hyphotesis)
                    self._track.append(state_post)
                else:
                    self._track.append(self._lag_assoc_buffer[-1]['prediction'])

        if self._track is None or self.deleter.delete_tracks(self.tracks):
            new_tracks = self.initiator.initiate(detections, time)
            if new_tracks:
                self._track = new_tracks.pop()
            else:
                self._track = None

        return time, self.tracks
