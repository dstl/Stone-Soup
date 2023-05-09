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
from stonesoup.types.update import Update
from stonesoup.updater.particle import ParticleUpdater


class SingleTargetExpectedLikelihoodParticleFilter(Tracker):
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
        self._track = None

    @property
    def tracks(self):
        return {self._track} if self._track else set()

    def __iter__(self):
        self.detector_iter = iter(self.detector)
        return super().__iter__()

    def __next__(self):
        time, detections = next(self.detector_iter)
        if self._track is not None:
            associations = self.data_associator.associate(
                self.tracks, detections, time)

            multihypothesis = associations[self._track]

            if len(multihypothesis) == 1 and not multihypothesis[0]:
                # if there is only one hypothesisa, and it is the missed detection hypothesis,
                # then there is nothing to update, so we just append the prediction
                new_state = multihypothesis[0].prediction
            else:
                # Compute new weights as the sum of the updated weights of the particles
                # multiplied by the weight of each hypothesis
                particle_weights_per_hypothesis = []
                for hypothesis in multihypothesis:
                    if not hypothesis:
                        particle_weights_per_hypothesis.append(
                            hypothesis.weight.log() + hypothesis.prediction.log_weight
                        )
                    else:
                        # Run the updater on the hypothesis
                        # NOTE: We MUST NOT resample here, as we do that after computing the
                        #       new weights
                        update = self.updater.update(hypothesis)
                        particle_weights_per_hypothesis.append(
                            hypothesis.weight.log() + update.log_weight
                        )

                # Sum the log weights of the particles
                new_log_weights = np.logaddexp.reduce(particle_weights_per_hypothesis, axis=0)

                # Normalise the weights
                new_log_weights -= logsumexp(new_log_weights)

                new_state = Update.from_state(
                    state=multihypothesis[0].prediction,
                    state_vector=multihypothesis[0].prediction.state_vector,
                    log_weight=new_log_weights,
                    hypothesis=multihypothesis,
                    timestamp=multihypothesis[0].measurement.timestamp
                )

                # Resample
                new_state = self.resampler.resample(new_state)

            self._track.append(new_state)

        if self._track is None or self.deleter.delete_tracks(self.tracks):
            new_tracks = self.initiator.initiate(detections, time)
            if new_tracks:
                self._track = new_tracks.pop()
            else:
                self._track = None

        return time, self.tracks


class MultiTargetExpectedLikelihoodParticleFilter(Tracker):
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
    initiator: Initiator = Property(doc="Initiator used to initialise the track.")
    deleter: Deleter = Property(doc="Initiator used to initialise the track.")
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
        if self.resampler is None:
            self.resampler = SystematicResampler()
        self._tracks = set()

    @property
    def tracks(self):
        return self._tracks

    def __iter__(self):
        self.detector_iter = iter(self.detector)
        return super().__iter__()

    def __next__(self):
        time, detections = next(self.detector_iter)

        timestamps = set([detection.timestamp for detection in detections])
        if len(timestamps) > 1:
            raise ValueError("All detections must have the same timestamp")

        # Perform data association
        associations = self.data_associator.associate(self.tracks, detections, time)

        unassociated_detections = set(detections)
        for track, multihypothesis in associations.items():

            if len(multihypothesis) == 1 and not multihypothesis[0]:
                # if there is only one hypothesisa, and it is the missed detection hypothesis,
                # then there is nothing to update, so we just append the prediction
                new_state = multihypothesis[0].prediction
            else:
                # Compute new weights as the sum of the updated weights of the particles
                # multiplied by the weight of each hypothesis
                particle_weights_per_hypothesis = []
                for hypothesis in multihypothesis:
                    if not hypothesis:
                        particle_weights_per_hypothesis.append(
                            hypothesis.weight.log() + hypothesis.prediction.log_weight
                        )
                    else:
                        # Run the updater on the hypothesis
                        # NOTE: We MUST NOT resample here, as we do that after computing the
                        #       new weights
                        update = self.updater.update(hypothesis)
                        particle_weights_per_hypothesis.append(
                            hypothesis.weight.log() + update.log_weight
                        )
                        unassociated_detections.discard(hypothesis.measurement)

                # Sum the log weights of the particles
                new_log_weights = np.logaddexp.reduce(particle_weights_per_hypothesis, axis=0)

                # Normalise the weights
                new_log_weights -= logsumexp(new_log_weights)

                new_state = Update.from_state(
                    state=multihypothesis[0].prediction,
                    state_vector=multihypothesis[0].prediction.state_vector,
                    log_weight=new_log_weights,
                    hypothesis=multihypothesis,
                    timestamp=multihypothesis[0].measurement.timestamp
                )

                # Resample
                new_state = self.resampler.resample(new_state)

            # Append the new state to the track
            track.append(new_state)

        self._tracks -= self.deleter.delete_tracks(self.tracks)
        self._tracks |= self.initiator.initiate(unassociated_detections, time)

        return time, self.tracks
