import numpy as np
from scipy.special import logsumexp

from stonesoup.base import Property
from stonesoup.dataassociator import DataAssociator
from stonesoup.deleter import Deleter
from stonesoup.initiator import Initiator
from stonesoup.reader import DetectionReader
from stonesoup.resampler import Resampler
from stonesoup.resampler.particle import SystematicResampler
from stonesoup.tracker import Tracker
from stonesoup.types.prediction import Prediction
from stonesoup.types.update import Update
from stonesoup.updater.particle import ParticleUpdater


class _BaseExpectedLikelihoodParticleFilter(Tracker):
    initiator: Initiator = Property(doc="Initiator used to initialise the track.")
    deleter: Deleter = Property(doc="Deleter used to delete the track")
    detector: DetectionReader = Property(doc="Detector used to generate detection objects.")
    data_associator: DataAssociator = Property(
        doc="Association algorithm to pair predictions to detections")
    updater: ParticleUpdater = Property(
        doc="Updater used to update the tracks. It is important that no resampling is performed "
            "in the updater, as this is handled by the tracker.")
    resampler: Resampler = Property(
        default=None,
        doc="Resampler used to resample the particles after the update step. If None, then "
            ":class:`~.SystematicResampler` is used.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.updater.resampler is not None:
            raise ValueError("Updater must not have a resampler")
        if self.resampler is None:
            self.resampler = SystematicResampler()

    def __iter__(self):
        self.detector_iter = iter(self.detector)
        return super().__iter__()

    def _get_detections(self):
        time, detections = next(self.detector_iter)
        timestamps = {detection.timestamp for detection in detections}
        if len(timestamps) > 1:
            raise ValueError("All detections must have the same timestamp")
        return time, detections

    def _get_new_state(self, multihypothesis):
        associated_detections = set()
        missed_detection_weight = next(hyp.weight for hyp in multihypothesis if not hyp)

        # Compute new weights as the sum of the updated weights of the particles
        # multiplied by the weight of each hypothesis
        particle_weights_per_hypothesis = []
        for hypothesis in multihypothesis:
            if not hypothesis:
                particle_weights_per_hypothesis.append(
                    np.log(hypothesis.weight) + hypothesis.prediction.log_weight
                )
            else:
                # Run the updater on the hypothesis
                # NOTE: We MUST NOT resample here, as we do that after computing the new weights
                update = self.updater.update(hypothesis)
                particle_weights_per_hypothesis.append(
                    np.log(hypothesis.weight) + update.log_weight
                )
                if hypothesis.weight > missed_detection_weight:
                    associated_detections.add(hypothesis.measurement)

        # Compute the new state
        new_log_weights = logsumexp(particle_weights_per_hypothesis, axis=0)

        # Normalise the weights
        new_log_weights -= logsumexp(new_log_weights)

        # Check if at least one reasonable measurement...
        if any(hypothesis.weight > missed_detection_weight for hypothesis in multihypothesis):
            # ...and if so use update type
            new_state = Update.from_state(
                state=multihypothesis[0].prediction,
                state_vector=multihypothesis[0].prediction.state_vector,
                log_weight=new_log_weights,
                hypothesis=multihypothesis,
                timestamp=multihypothesis[0].measurement.timestamp
            )
        else:
            # ...and if not, treat as a prediction
            new_state = Prediction.from_state(
                state=multihypothesis[0].prediction,
                state_vector=multihypothesis[0].prediction.state_vector,
                log_weight=new_log_weights,
                timestamp=multihypothesis[0].measurement.timestamp
            )

        # Resample
        new_state = self.resampler.resample(new_state)

        return new_state, associated_detections


class SingleTargetExpectedLikelihoodParticleFilter(_BaseExpectedLikelihoodParticleFilter):
    """An expected likelihood particle filter (ELPF) [#]_ for tracking a single target.

    Track a single object using Stone Soup components. The tracker works by
    first calling the :attr:`data_associator` with the active track, and then
    either updating the track state with the result of the :attr:`updater` if
    a detection is associated, or with the prediction if no detection is
    associated to the track. The track is then checked for deletion by the
    :attr:`deleter`, and if deleted the :attr:`initiator` is called to generate
    a new track. Similarly, if no track is present (i.e. tracker is initialised
    or deleted in previous iteration), only the :attr:`initiator` is called.

    Parameters
    ----------

    Attributes
    ----------
    track : :class:`~.Track`
        Current track being maintained. Also accessible as the sole item in
        :attr:`tracks`

    References
    ----------
    .. [#] Marrs, A., Maskell, S., and Bar-Shalom, Y., “Expected likelihood for tracking in clutter
       with particle filters”, in Signal and Data Processing of Small Targets 2002, 2002,
       vol. 4728, pp. 230–239. doi:10.1117/12.478507.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._track = None

    @property
    def tracks(self):
        return {self._track} if self._track else set()

    def __next__(self):
        time, detections = self._get_detections()

        if self._track is not None:
            # Perform data association
            associations = self.data_associator.associate(self.tracks, detections, time)
            multihypothesis = associations[self._track]

            # Update the track
            new_state, _ = self._get_new_state(multihypothesis)
            self._track.append(new_state)

        # Track initiation/deletion
        if self._track is None or self.deleter.delete_tracks(self.tracks):
            new_tracks = self.initiator.initiate(detections, time)
            if new_tracks:
                self._track = new_tracks.pop()
            else:
                self._track = None

        return time, self.tracks


class MultiTargetExpectedLikelihoodParticleFilter(_BaseExpectedLikelihoodParticleFilter):
    """An expected likelihood particle filter (ELPF) [#]_ for tracking multiple targets.

    Track multiple objects using Stone Soup components. The tracker works by
    first calling the :attr:`data_associator` with the active tracks, and then
    either updating the track state with the result of the
    :attr:`data_associator` that reduces the (Particle) Mixture of all
    possible track-detection associations, or with the prediction if no
    detection is associated to the track. Tracks are then checked for deletion
    by the :attr:`deleter`, and remaining unassociated detections are passed
    to the :attr:`initiator` to generate new tracks.

    Parameters
    ----------

    References
    ----------
    .. [#] Marrs, A., Maskell, S., and Bar-Shalom, Y., “Expected likelihood for tracking in clutter
       with particle filters”, in Signal and Data Processing of Small Targets 2002, 2002,
       vol. 4728, pp. 230–239. doi:10.1117/12.478507.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tracks = set()

    @property
    def tracks(self):
        return self._tracks

    def __next__(self):
        time, detections = self._get_detections()

        # Perform data association
        associations = self.data_associator.associate(self.tracks, detections, time)

        unassociated_detections = set(detections)
        for track, multihypothesis in associations.items():

            # Update the track
            new_state, associated_detections = self._get_new_state(multihypothesis)
            track.append(new_state)

            # Remove associated detections from the set of unassociated detections
            unassociated_detections -= associated_detections

        # Initiate new tracks and delete old tracks
        self._tracks -= self.deleter.delete_tracks(self.tracks)
        self._tracks |= self.initiator.initiate(unassociated_detections, time)

        return time, self.tracks
