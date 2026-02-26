import numpy as np
from scipy.special import logsumexp
from stonesoup.base import Property
from stonesoup.tracker.base import Tracker, _TrackerMixInNext
from stonesoup.types.track import Track
from stonesoup.types.hypothesis import SingleHypothesis


class _BaseFixedLagTracker(_TrackerMixInNext, Tracker):

    initiator = Property('Initiator used to initialise the track')
    deleter = Property('Deleter used to delete the track')
    detector = Property('Detector to generate detections')
#    data_associator = Property('Data associator to associate detections to tracks')
    predictor = Property('Predictor to predict the next state')
    updater = Property('Updater to update the tracks with the measurements')
    lag = Property('The lag in time steps',
                   default=3)

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


class FixedLagTracker(_BaseFixedLagTracker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._track = None  # current track
        self._lag_assoc_buffer = []  # associations
        self.lag_times = []  # to store the lag times to reconstruct the states

    @property
    def tracks(self) -> set[Track]:
        return {self._track} if self._track else set()

    def _fill_lag_buffer(self, associations):
        """
            Fill the lag storage with associations
        """
        self._lag_assoc_buffer.append(associations)

        if associations['detections']:
            for det in associations['detections']:
                hyphotesis = SingleHypothesis(prediction=associations['prediction'],
                                              measurement=det)  # assuming single measurement for simplicity
            state_post = self.updater.update(hyphotesis)  # assuming single measurement for simplicity
            self._track.append(state_post)
        else:
            self._track.append(associations['prediction'])

    def _process_lag(self):
        """
        Main function to process the lag, update the states and weights in the lag buffer and then update the track with the new measurement
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

        new_weights = (new_weights_lag[self._track]['oldweights'] + new_weights_lag[self._track]['logposterior'] -
                       new_weights_lag[self._track]['logprior'])

        # normalise the weights
        new_weights -= logsumexp(new_weights)

        # now update the weights at the lag position
        self._track[len(self._track) - (self.lag + 1)].log_weight = new_weights

        # in case do the resample
        self._track[len(self._track) - (self.lag + 1)] = self.updater.resampler.resample(self._track[len(self._track) - (self.lag + 1)])

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
                                                      measurement=det)  # assuming single measurement for simplicity
                    state_post = self.updater.update(hyphotesis)  # assuming single detection for simplicity
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
                        hyphotesis = SingleHypothesis(prediction=self._lag_assoc_buffer[-1]['prediction'],
                                                      measurement=det)  # assuming single measurement for simplicity
                    state_post = self.updater.update(hyphotesis)  # assuming single detection for simplicity
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
