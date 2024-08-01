import numpy as np
from scipy.special import logsumexp

from stonesoup.base import Property
from stonesoup.hypothesiser.simple import SimpleHypothesiser
from stonesoup.initiator.base import ParticleInitiator
from stonesoup.predictor.particle import SMCPHDPredictor
from stonesoup.resampler import Resampler
from stonesoup.types.multihypothesis import MultipleHypothesis
from stonesoup.types.state import ParticleState
from stonesoup.types.track import Track
from stonesoup.types.update import Update
from stonesoup.updater.particle import SMCPHDUpdater


class SMCPHDInitiator(ParticleInitiator):
    r"""Sequential Monte Carlo Probability Hypothesis Density (SMC-PHD) Initiator class

    An implementation of a particle initiator that uses a Sequential Monte Carlo Probability
    Hypothesis Density (SMC-PHD) filter to generate tracks from detections, based on [#phdi]_.

    Note
    ----
    The current implementation does not support non-association weights (i.e. :math:`\rho_i` in
    [#phdi]_).

    Parameters
    ----------

    References
    ----------
    .. [#phdi] P. Horridge and S. Maskell,  “Using a probabilistic hypothesis density filter to
           confirm tracks in a multi-target environment,” in 2011 Jahrestagung der Gesellschaft
           fr Informatik, October 2011.
    """

    prior_state: ParticleState = Property(
        doc="Prior particle state used to initialise the PHD density")
    predictor: SMCPHDPredictor = Property(doc="SMC-PHD predictor used to predict the PHD density")
    updater: SMCPHDUpdater = Property(doc="SMC-PHD updater used to update the PHD density")
    threshold: float = Property(
        default=0.9,
        doc="Existence probability threshold for initiating tracks. Default is 0.9")
    num_track_samples: int = Property(
        default=None,
        doc="Number of particles for initiated tracks. Default is None in which case the "
            "number of particles will be set to the number of particles in the prior state.")
    resampler: Resampler = Property(
        default=None,
        doc='Resampler used to resample the particles of output tracks before returning. Default '
            'is None in which case the resampler of the updater will be used.')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.resampler is None:
            if self.updater.resampler is None:
                raise ValueError("No resampler specified and updater has no resampler")
            self.resampler = self.updater.resampler
        self._state = self.prior_state
        self._hypothesiser = SimpleHypothesiser(self.predictor)

    def initiate(self, detections, timestamp, **kwargs):
        tracks = set()

        # Hypothesise
        hypotheses = self._hypothesiser.hypothesise(self._state, detections, timestamp)

        # Sort hypotheses, so that missed detection is always first
        hypotheses.single_hypotheses.sort(key=bool)

        # Calculate weights per hypothesis
        log_weights_per_hyp = self.updater.get_log_weights_per_hypothesis(hypotheses)

        # Calculate intensity per hypothesis
        log_intensity_per_hyp = logsumexp(log_weights_per_hyp, axis=0)

        # Find hypotheses with intensity above threshold and initiate
        valid_hyp_inds = np.flatnonzero(np.exp(log_intensity_per_hyp) > self.threshold).astype(int)
        for idx in valid_hyp_inds:

            # Get hypothesis
            hypothesis = hypotheses.single_hypotheses[idx]

            # Skip missed detection
            if not hypothesis:
                continue

            # Create state update
            state = Update.from_state(state=hypothesis.prediction,
                                      hypothesis=hypothesis,
                                      timestamp=timestamp)
            # Normalise weights
            state.log_weight = log_weights_per_hyp[:, idx] - log_intensity_per_hyp[idx]

            # Resample particles
            state = self.resampler.resample(state, self.num_track_samples)

            # Create track
            tracks.add(Track([state]))

        # Filter out hypotheses for detections above threshold (always keep missed detection)
        remaining_hyp_inds = (set(range(len(hypotheses))) - set(valid_hyp_inds)) | {0}
        remaining_hypotheses = MultipleHypothesis([hypotheses[i] for i in remaining_hyp_inds])

        # Update PHD state
        self._state = self.updater.update(remaining_hypotheses)

        return tracks
