# -*- coding: utf-8 -*-
import numpy as np

from .base import Tracker
from ..base import Property
from ..dataassociator import DataAssociator
from ..deleter import Deleter
from ..reader import DetectionReader
from ..initiator import Initiator
from ..updater import Updater
from ..types.array import StateVectors
from ..types.prediction import GaussianStatePrediction
from ..types.update import GaussianStateUpdate
from ..functions import gm_reduce_single
from stonesoup.buffered_generator import BufferedGenerator


class SingleTargetTracker(Tracker):
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
    initiator = Property(
        Initiator,
        doc="Initiator used to initialise the track.")
    deleter = Property(
        Deleter,
        doc="Deleter used to delete the track")
    detector = Property(
        DetectionReader,
        doc="Detector used to generate detection objects.")
    data_associator = Property(
        DataAssociator,
        doc="Association algorithm to pair predictions to detections")
    updater = Property(
        Updater,
        doc="Updater used to update the track object to the new state.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @BufferedGenerator.generator_method
    def tracks_gen(self):
        track = None
        for time, detections in self.detector:
            if track is not None:
                associations = self.data_associator.associate(
                    {track}, detections, time)
                if associations[track]:
                    state_post = self.updater.update(associations[track])
                    track.append(state_post)
                else:
                    track.append(
                        associations[track].prediction)

            if track is None or self.deleter.delete_tracks({track}):
                new_tracks = self.initiator.initiate(detections)
                if new_tracks:
                    track = new_tracks.pop()
                else:
                    track = None

            yield (time, {track}) if track is not None else (time, set())


class MultiTargetTracker(Tracker):
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
    initiator = Property(
        Initiator,
        doc="Initiator used to initialise the track.")
    deleter = Property(
        Deleter,
        doc="Initiator used to initialise the track.")
    detector = Property(
        DetectionReader,
        doc="Detector used to generate detection objects.")
    data_associator = Property(
        DataAssociator,
        doc="Association algorithm to pair predictions to detections")
    updater = Property(
        Updater,
        doc="Updater used to update the track object to the new state.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @BufferedGenerator.generator_method
    def tracks_gen(self):
        tracks = set()

        for time, detections in self.detector:

            associations = self.data_associator.associate(
                tracks, detections, time)
            associated_detections = set()
            for track, hypothesis in associations.items():
                if hypothesis:
                    state_post = self.updater.update(hypothesis)
                    track.append(state_post)
                    associated_detections.add(hypothesis.measurement)
                else:
                    track.append(hypothesis.prediction)

            tracks -= self.deleter.delete_tracks(tracks)
            tracks |= self.initiator.initiate(
                detections - associated_detections)

            yield time, tracks


class MultiTargetMixtureTracker(Tracker):
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
    initiator = Property(
        Initiator,
        doc="Initiator used to initialise the track.")
    deleter = Property(
        Deleter,
        doc="Initiator used to initialise the track.")
    detector = Property(
        DetectionReader,
        doc="Detector used to generate detection objects.")
    data_associator = Property(
        DataAssociator,
        doc="Association algorithm to pair predictions to detections")
    updater = Property(
        Updater,
        doc="Updater used to update the track object to the new state.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @BufferedGenerator.generator_method
    def tracks_gen(self):
        tracks = set()

        for time, detections in self.detector:

            associations = self.data_associator.associate(
                tracks, detections, time)
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

            tracks -= self.deleter.delete_tracks(tracks)
            tracks |= self.initiator.initiate(
                unassociated_detections)

            yield time, tracks
