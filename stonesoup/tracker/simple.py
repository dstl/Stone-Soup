import datetime

import numpy as np

from .base import Tracker, _TrackerMixInNext
from ..base import Property
from ..dataassociator import DataAssociator
from ..deleter import Deleter
from ..functions import gm_reduce_single
from ..initiator import Initiator
from ..reader import DetectionReader
from ..types.array import StateVectors
from ..types.prediction import GaussianStatePrediction
from ..types.track import Track
from ..types.update import GaussianStateUpdate
from ..types.state import ParticleState
from ..updater import Updater
from ..predictor.particle import ParticlePredictor


class SingleTargetTracker(_TrackerMixInNext, Tracker):
    """A simple single target tracker.

    Track a single object using Stone Soup components. The tracker works by
    first calling the :attr:`data_associator` with the active track, and then
    either updating the track state with the result of the :attr:`updater` if
    a detection is associated, or with the prediction if no detection is
    associated to the track. The track is then checked for deletion by the
    :attr:`deleter`, and if deleted the :attr:`initiator` is called to generate
    a new track. Similarly if no track is present (i.e. tracker is initialised
    or deleted in previous iteration), only the :attr:`initiator` is called.

    Parameters
    ----------

    Attributes
    ----------
    track : :class:`~.Track`
        Current track being maintained. Also accessible as the sole item in
        :attr:`tracks`
    """
    initiator: Initiator = Property(doc="Initiator used to initialise the track.")
    deleter: Deleter = Property(doc="Deleter used to delete tracks.")
    detector: DetectionReader = Property(doc="Detector used to generate detection objects.")
    data_associator: DataAssociator = Property(
        doc="Association algorithm to pair predictions to detections")
    updater: Updater = Property(doc="Updater used to update the track object to the new state.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._track = None

    @property
    def tracks(self) -> set[Track]:
        return {self._track} if self._track else set()

    def __next__(self) -> tuple[datetime.datetime, set[Track]]:
        time, detections = next(self.detector_iter)
        if self._track is not None:
            associations = self.data_associator.associate(
                self.tracks, detections, time)
            if associations[self._track]:
                state_post = self.updater.update(associations[self._track])
                self._track.append(state_post)
            else:
                self._track.append(
                    associations[self._track].prediction)

        if self._track is None or self.deleter.delete_tracks(self.tracks):
            new_tracks = self.initiator.initiate(detections, time)
            if new_tracks:
                self._track = new_tracks.pop()
            else:
                self._track = None

        return time, self.tracks


class SingleTargetMixtureTracker(_TrackerMixInNext, Tracker):
    """ A simple single target tracking that receives associations from a
    (Gaussian) Mixture associator.

    Track single objects using Stone Soup components. The tracker works by
    first calling the :attr:`data_associator` with the active track, and then
    either updating the track state with the result of the
    :attr:`data_associator` that reduces the (Gaussian) Mixture of all
    possible track-detection associations, or with the prediction if no
    detection is associated to the track.
    The track is then checked for deletion
    by the :attr:`deleter`, and remaining unassociated detections are passed
    to the :attr:`initiator` to generate new track.

    Parameters
    ----------
    """
    initiator: Initiator = Property(doc="Initiator used to initialise the track.")
    deleter: Deleter = Property(doc="Deleter used to delete tracks.")
    detector: DetectionReader = Property(doc="Detector used to generate detection objects.")
    data_associator: DataAssociator = Property(
        doc="Association algorithm to pair predictions to detections")
    updater: Updater = Property(doc="Updater used to update the track object to the new state.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._track = None

    @property
    def tracks(self) -> set[Track]:
        return {self._track} if self._track else set()

    def __next__(self) -> tuple[datetime.datetime, set[Track]]:
        time, detections = next(self.detector_iter)

        if self._track is not None:
            associations = self.data_associator.associate(
                self.tracks, detections, time)

            unassociated_detections = set(detections)
            for track, multihypothesis in associations.items():

                # calculate the Track's state as a Gaussian Mixture of
                # its possible associations with each detection, then
                # reduce the Mixture to a single Gaussian State
                posterior_states = []
                posterior_state_weights = []
                for hypothesis in multihypothesis:
                    if not hypothesis:
                        posterior_states.append(hypothesis.prediction)
                    else:
                        posterior_states.append(
                            self.updater.update(hypothesis))
                    posterior_state_weights.append(
                        hypothesis.probability)

                means = StateVectors([state.state_vector for state in posterior_states])
                covars = np.stack([state.covar for state in posterior_states], axis=2)
                weights = np.asarray(posterior_state_weights)

                # Reduce the mixture of states to one posterior estimate Gaussian
                post_mean, post_covar = gm_reduce_single(means, covars, weights)

                missed_detection_weight = next(hyp.weight for hyp in multihypothesis if not hyp)

                # Check if at least one reasonable measurement...
                if any(hypothesis.weight > missed_detection_weight
                       for hypothesis in multihypothesis):
                    # ...and if so use update type
                    track.append(GaussianStateUpdate(
                        post_mean, post_covar,
                        multihypothesis,
                        multihypothesis[0].measurement.timestamp))
                else:
                    # ...and if not, treat as a prediction
                    track.append(GaussianStatePrediction(
                        post_mean, post_covar,
                        multihypothesis[0].prediction.timestamp))

                # any detections in multihypothesis that had an
                # association score (weight) lower than or equal to the
                # association score of "MissedDetection" is considered
                # unassociated - candidate for initiating a new Track
                for hyp in multihypothesis:
                    if hyp.weight > missed_detection_weight:
                        if hyp.measurement in unassociated_detections:
                            unassociated_detections.remove(hyp.measurement)

        if self._track is None or self.deleter.delete_tracks(self.tracks):
            new_tracks = self.initiator.initiate(detections, time)
            if new_tracks:
                self._track = new_tracks.pop()
            else:
                self._track = None

        return time, self.tracks


class MultiTargetTracker(_TrackerMixInNext, Tracker):
    """A simple multi target tracker.

    Track multiple objects using Stone Soup components. The tracker works by
    first calling the :attr:`data_associator` with the active tracks, and then
    either updating the track state with the result of the :attr:`updater` if
    a detection is associated, or with the prediction if no detection is
    associated to the track. Tracks are then checked for deletion by the
    :attr:`deleter`, and remaining unassociated detections are passed to the
    :attr:`initiator` to generate new tracks.

    Parameters
    ----------
    """
    initiator: Initiator = Property(doc="Initiator used to initialise the track.")
    deleter: Deleter = Property(doc="Deleter used to delete tracks.")
    detector: DetectionReader = Property(doc="Detector used to generate detection objects.")
    data_associator: DataAssociator = Property(
        doc="Association algorithm to pair predictions to detections")
    updater: Updater = Property(doc="Updater used to update the track object to the new state.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tracks = set()

    @property
    def tracks(self) -> set[Track]:
        return self._tracks

    def __next__(self) -> tuple[datetime.datetime, set[Track]]:
        time, detections = next(self.detector_iter)

        associations = self.data_associator.associate(
            self.tracks, detections, time)
        associated_detections = set()
        for track, hypothesis in associations.items():
            if hypothesis:
                state_post = self.updater.update(hypothesis)
                track.append(state_post)
                associated_detections.add(hypothesis.measurement)
            else:
                track.append(hypothesis.prediction)

        self._tracks -= self.deleter.delete_tracks(self.tracks)
        self._tracks |= self.initiator.initiate(
            detections - associated_detections, time)

        return time, self.tracks


class MultiTargetMixtureTracker(_TrackerMixInNext, Tracker):
    """A simple multi target tracker that receives associations from a
    (Gaussian) Mixture associator.

    Track multiple objects using Stone Soup components. The tracker works by
    first calling the :attr:`data_associator` with the active tracks, and then
    either updating the track state with the result of the
    :attr:`data_associator` that reduces the (Gaussian) Mixture of all
    possible track-detection associations, or with the prediction if no
    detection is associated to the track. Tracks are then checked for deletion
    by the :attr:`deleter`, and remaining unassociated detections are passed
    to the :attr:`initiator` to generate new tracks.

    Parameters
    ----------
    """
    initiator: Initiator = Property(doc="Initiator used to initialise the track.")
    deleter: Deleter = Property(doc="Deleter used to delete tracks.")
    detector: DetectionReader = Property(doc="Detector used to generate detection objects.")
    data_associator: DataAssociator = Property(
        doc="Association algorithm to pair predictions to detections")
    updater: Updater = Property(doc="Updater used to update the track object to the new state.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tracks = set()

    @property
    def tracks(self) -> set[Track]:
        return self._tracks

    def __next__(self) -> tuple[datetime.datetime, set[Track]]:
        time, detections = next(self.detector_iter)

        associations = self.data_associator.associate(
            self.tracks, detections, time)
        unassociated_detections = set(detections)
        for track, multihypothesis in associations.items():

            # calculate each Track's state as a Gaussian Mixture of
            # its possible associations with each detection, then
            # reduce the Mixture to a single Gaussian State
            posterior_states = []
            posterior_state_weights = []
            for hypothesis in multihypothesis:
                if not hypothesis:
                    posterior_states.append(hypothesis.prediction)
                else:
                    posterior_states.append(
                        self.updater.update(hypothesis))
                posterior_state_weights.append(
                    hypothesis.probability)

            means = StateVectors([state.state_vector for state in posterior_states])
            covars = np.stack([state.covar for state in posterior_states], axis=2)
            weights = np.asarray(posterior_state_weights)

            post_mean, post_covar = gm_reduce_single(means, covars, weights)

            missed_detection_weight = next(hyp.weight for hyp in multihypothesis if not hyp)

            # Check if at least one reasonable measurement...
            if any(hypothesis.weight > missed_detection_weight
                   for hypothesis in multihypothesis):
                # ...and if so use update type
                track.append(GaussianStateUpdate(
                    post_mean, post_covar,
                    multihypothesis,
                    multihypothesis[0].measurement.timestamp))
            else:
                # ...and if not, treat as a prediction
                track.append(GaussianStatePrediction(
                    post_mean, post_covar,
                    multihypothesis[0].prediction.timestamp))

            # any detections in multihypothesis that had an
            # association score (weight) lower than or equal to the
            # association score of "MissedDetection" is considered
            # unassociated - candidate for initiating a new Track
            for hyp in multihypothesis:
                if hyp.weight > missed_detection_weight:
                    if hyp.measurement in unassociated_detections:
                        unassociated_detections.remove(hyp.measurement)

        self._tracks -= self.deleter.delete_tracks(self.tracks)
        self._tracks |= self.initiator.initiate(
            unassociated_detections, time)

        return time, self.tracks


class BayesianSearchMultiTargetTracker(MultiTargetTracker):

    search_state: ParticleState = Property(
        doc="State object which stores particles representing grid cells "
            "and associated search weights")

    detection_probability: float = Property(
        doc="Deteciton probability used to update search weights.")

    search_predictor: ParticlePredictor = Property(
        doc="Predictor used to predict the particle distribution"
        "over the unknown target distribution")


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._log_det_prob = np.log(self.detection_probability)

    @property
    def platforms(self):
        return getattr(self.detector, "platforms", set())

    @property
    def sensors(self):
        sensor_set = set()
        for platform in self.platforms:
            sensor_set.update(platform.sensors)

        sensor_set.update(getattr(self.detector, "sensors", set()))

        return sensor_set

    def __next__(self):
        time, _ = super(BayesianSearchMultiTargetTracker, self).__next__()

        # Predict over undetected target distribution
        prediction = self.search_predictor.predict(self.search_state,
                                                   timestamp=time)

        # Update particles using log maths and log weights
        update = ParticleState.from_state(prediction)
        for sensor in self.sensors:

            # Get visible particles for the sensor in question
            vis_parts = sensor.is_detectable(update)

            # Calculate the multiplier for incrementing the existence probability for all other
            # particles.
            non_vis_multiplier = \
                np.sum(-1*np.log1p(-np.exp(update.log_weight[vis_parts]+self._log_det_prob)))

            update.log_weight += non_vis_multiplier

            # Decrease existence probability for particles in the FOV
            update.log_weight[vis_parts] += np.log1p(-self.detection_probability)

        self.search_state = update

        return time, self.tracks
