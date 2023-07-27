import numpy as np

from ..types.hypothesis import MultiHypothesis
from ..types.tracklet import Scan
from ...base import Base, Property
from ...dataassociator.probability import JPDA
from ...tracker import Tracker
from ...predictor import Predictor
from ...types.mixture import GaussianMixture
from ...types.multihypothesis import MultipleHypothesis
from ...updater import Updater
from ...dataassociator import DataAssociator
from ...types.numeric import Probability
from ...types.prediction import Prediction
from ...types.array import StateVectors
from ...types.update import Update
from ...initiator import Initiator
from ...functions import gm_reduce_single


from ..reader.tracklet import PseudoMeasExtractor, TrackletExtractor
from ..types.update import TwoStateGaussianStateUpdate


class _BaseFuseTracker(Base):
    initiator: Initiator = Property(doc='The initiator used to initiate fused tracks')
    predictor: Predictor = Property(doc='Predictor used to predict fused tracks')
    updater: Updater = Property(doc='Updater used to update fused tracks')
    associator: DataAssociator = Property(doc='Associator used to associate fused tracks with'
                                              'pseudomeasurements')
    death_rate: float = Property(doc='The exponential death rate of tracks. Default is 1e-4',
                                 default=1e-4)
    prob_detect: Probability = Property(doc='The probability of detection', default=0.9)
    delete_thresh: Probability = Property(doc='The existence probability deletion threshold',
                                          default=0.1)

    def __init__(self, *args, **kwargs):
        super(_BaseFuseTracker, self).__init__(*args, **kwargs)
        self._max_track_id = 0

    def process_scan(self, scan, tracks, current_end_time):
        new_start_time = scan.start_time
        new_end_time = scan.end_time
        if current_end_time and new_start_time < current_end_time:
            print('Scans out of order! Skipping a scan...')
            return tracks, current_end_time

        if hasattr(self.initiator, 'predict'):
            self.initiator.predict(new_start_time, new_end_time)
            self.initiator.current_end_time = new_end_time

        # Predict two-state tracks forward
        for track in tracks:
            self.predict_track(track, current_end_time, new_start_time, new_end_time,
                               self.death_rate)

        current_start_time = new_start_time
        current_end_time = new_end_time

        if not len(scan.sensor_scans):
            tracks = list(tracks)
            detections = set()

            # Perform data association
            associations = self.associator.associate(tracks, detections,
                                                     timestamp=current_end_time)
            # Update tracks
            for track in tracks:
                self.update_track(track, associations[track], scan.id)

            # Initiate new tracks on unassociated detections
            if isinstance(self.associator, JPDA):
                assoc_detections = set(
                    [h.measurement for hyp in associations.values() for h in hyp if h])
            else:
                assoc_detections = set(
                    [hyp.measurement for hyp in associations.values() if hyp])

            tracks = set(tracks)

        for sensor_scan in scan.sensor_scans:
            tracks = list(tracks)
            detections = set(sensor_scan.detections)

            # Perform data association
            associations = self.associator.associate(tracks, detections,
                                                     timestamp=current_end_time)
            # Update tracks
            for track in tracks:
                self.update_track(track, associations[track], scan.id)

            # Initiate new tracks on unassociated detections
            if isinstance(self.associator, JPDA):
                assoc_detections = set(
                    [h.measurement for hyp in associations.values() for h in hyp if h])
            else:
                assoc_detections = set(
                    [hyp.measurement for hyp in associations.values() if hyp])


            tracks = set(tracks)
            unassoc_detections = set(detections) - assoc_detections
            if isinstance(sensor_scan.sensor_id, str):
                tracks |= self.initiator.initiate(unassoc_detections, sensor_scan.timestamp,
                                                  sensor_scan.timestamp,
                                                  sensor_id=sensor_scan.sensor_id)
            else:
                tracks |= self.initiator.initiate(unassoc_detections, current_start_time,
                                                  current_end_time, sensor_id=sensor_scan.sensor_id)
        try:
            self.initiator.current_end_time = current_end_time
        except AttributeError:
            pass

        tracks -= self.delete_tracks(tracks)
        return tracks, current_end_time

    def predict_track(self, track, current_end_time, new_start_time, new_end_time,
                      death_rate=0.):

        # Predict existence
        survive_prob = np.exp(-death_rate * (new_end_time - current_end_time).total_seconds())
        track.exist_prob = track.exist_prob * survive_prob

        # Predict forward
        # p(x_k, x_{k+\Delta} | y^{1:S}_{1:k})
        if not isinstance(track.state, GaussianMixture):
            prediction = self.predictor.predict(track.state, current_end_time=current_end_time,
                                                new_start_time=new_start_time,
                                                new_end_time=new_end_time)
        else:
            pred_components = []
            for component in track.state:
                pred_components.append(self.predictor.predict(component,
                                                              current_end_time=current_end_time,
                                                              new_start_time=new_start_time,
                                                              new_end_time=new_end_time))
            prediction = GaussianMixture(pred_components)
        # Append prediction to track history
        track.append(prediction)

    def update_track(self, track, hypothesis, scan_id):
        last_state = track.states[-1]

        if isinstance(self.associator, JPDA):
            # calculate each Track's state as a Gaussian Mixture of
            # its possible associations with each detection, then
            # reduce the Mixture to a single Gaussian State
            posterior_states = []
            posterior_state_weights = []
            for hyp in hypothesis:
                if not hyp:
                    posterior_states.append(hyp.prediction)
                    # Ensure null hyp weight is at index 0
                    posterior_state_weights.insert(0, hyp.probability)
                else:
                    posterior_states.append(
                        self.updater.update(hyp))
                    posterior_state_weights.append(
                        hyp.probability)
                    if 'track_id' in hyp.measurement.metadata:
                        try:
                            track.track_ids.add(hyp.measurement.metadata['track_id'])
                        except AttributeError:
                            track.track_ids = {hyp.measurement.metadata['track_id']}

            means = StateVectors([state.state_vector for state in posterior_states])
            covars = np.stack([state.covar for state in posterior_states], axis=2)
            weights = np.asarray(posterior_state_weights)

            post_mean, post_covar = gm_reduce_single(means, covars, weights)

            update = TwoStateGaussianStateUpdate(post_mean, post_covar,
                                                 start_time=posterior_states[0].start_time,
                                                 end_time=posterior_states[0].end_time,
                                                 hypothesis=hypothesis)
            track[-1] = update
            # Compute existence probability
            non_exist_weight = 1 - track.exist_prob
            non_det_weight = (1 - self.prob_detect) * track.exist_prob
            null_exist_weight = non_det_weight / (non_exist_weight + non_det_weight)
            exist_probs = np.array([null_exist_weight, *[1. for i in range(len(weights) - 1)]])
            track.exist_prob = Probability.sum(exist_probs * weights)
        else:
            if hypothesis:
                # Perform update using the hypothesis
                update = self.updater.update(hypothesis)
                # Modify track states depending on type of last state
                if isinstance(last_state, Update) and last_state.timestamp == update.timestamp:
                    # If the last scan was an update with the same timestamp, we need to modify this
                    # state to reflect the computed mean and covariance, as well as the hypotheses that
                    # resulted to this
                    hyp = last_state.hypothesis
                    try:
                        hyp.measurements.append(hypothesis.measurement)
                    except AttributeError:
                        hyp = MultiHypothesis(prediction=hypothesis.prediction,
                                              measurements=[hyp.measurement,
                                                            hypothesis.measurement])
                    update.hypothesis = hyp  # Update the hypothesis
                    track[-1] = update  # Replace the last state
                elif isinstance(last_state,
                                Prediction) and last_state.timestamp == update.timestamp:
                    # If the last state was a prediction with the same timestamp, it means that the
                    # state was created by a sensor scan in the same overall scan, due to the track not
                    # having been associated to any measurement. Therefore, we replace the prediction
                    # with the update
                    update.hypothesis = MultiHypothesis(prediction=hypothesis.prediction,
                                                        measurements=[hypothesis.measurement])
                    track[-1] = update
                else:
                    # Else simply append the update to the track history
                    update.hypothesis = MultiHypothesis(prediction=hypothesis.prediction,
                                                        measurements=[hypothesis.measurement])
                    track.append(update)
                # Set existence probability to 1
                track.exist_prob = 1
                if 'track_id' in hypothesis.measurement.metadata:
                    try:
                        track.track_ids.add(hypothesis.measurement.metadata['track_id'])
                    except AttributeError:
                        track.track_ids = {hypothesis.measurement.metadata['track_id']}
            else:
                # If the track was not associated to any measurement, simply update the existence
                # probability
                non_exist_weight = 1 - track.exist_prob
                non_det_weight = (1 - self.prob_detect) * track.exist_prob
                track.exist_prob = non_det_weight / (non_exist_weight + non_det_weight)

    def delete_tracks(self, tracks):
        del_tracks = set([track for track in tracks if track.exist_prob < self.delete_thresh])
        return del_tracks


class FuseTracker(Tracker, _BaseFuseTracker):
    """

    """

    detector: PseudoMeasExtractor = Property(doc='The pseudo-measurement extractor')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tracks = set()
        self._current_end_time = None

    @property
    def tracks(self):
        return self._tracks

    def __iter__(self):
        self.detector_iter = iter(self.detector)
        return super().__iter__()

    def __next__(self):
        timestamp, scans = next(self.detector_iter)
        for scan in scans:
            self._tracks, self._current_end_time = self.process_scan(scan, self.tracks, self._current_end_time)
        return timestamp, self.tracks


class FuseTracker2(_BaseFuseTracker):

    tracklet_extractor: TrackletExtractor = Property(doc='The tracklet extractor')
    pseudomeas_extractor: PseudoMeasExtractor = Property(doc='The pseudo-measurement extractor')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tracks = set()
        self._current_end_time = None

    @property
    def tracks(self):
        return self._tracks

    def process_tracks(self, alltracks, timestamp):
        # Extract tracklets
        tracklets = self.tracklet_extractor.extract(alltracks, timestamp)
        # Extract pseudo-measurements
        scans = self.pseudomeas_extractor.extract(tracklets, timestamp)
        if not len(scans) and self._current_end_time and timestamp - self._current_end_time >= self.tracklet_extractor.fuse_interval:
            scans = [Scan(self._current_end_time, timestamp, [])]

        for scan in scans:
            self._tracks, self._current_end_time = self.process_scan(scan, self.tracks,
                                                                     self._current_end_time)
        return self.tracks

    def process_scans(self, scans):
        for scan in scans:
            self._tracks, self._current_end_time = self.process_scan(scan, self.tracks,
                                                                     self._current_end_time)
        return self.tracks


