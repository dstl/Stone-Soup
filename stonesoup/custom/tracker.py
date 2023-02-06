from abc import abstractmethod
from copy import copy
from datetime import datetime, timezone

import numpy as np
from scipy.stats import multivariate_normal

from stonesoup.base import Property, Base
from stonesoup.custom.dataassociator.jipda import JIPDAWithEHM2
from stonesoup.custom.initiator.smcphd import SMCPHDFilter, SMCPHDInitiator, ISMCPHDFilter, \
    ISMCPHDInitiator
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.functions import gm_reduce_single
from stonesoup.gater.distance import DistanceGater
from stonesoup.custom.hypothesiser.probability import PDAHypothesiser, IPDAHypothesiser
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.models.measurement import MeasurementModel
from stonesoup.models.transition import TransitionModel
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.resampler.particle import SystematicResampler
from stonesoup.types.array import StateVectors
from stonesoup.types.mixture import GaussianMixture
from stonesoup.types.numeric import Probability
from stonesoup.types.state import State, ParticleState
from stonesoup.types.update import GaussianStateUpdate
from stonesoup.updater.kalman import KalmanUpdater


class _BaseTracker(Base):
    transition_model: TransitionModel = Property(doc='The transition model')
    measurement_model: MeasurementModel = Property(doc='The measurement model')
    prob_detection: Probability = Property(doc='The probability of detection')
    prob_death: Probability = Property(doc='The probability of death')
    prob_birth: Probability = Property(doc='The probability of birth')
    birth_rate: float = Property(
        doc='The birth rate (i.e. number of new/born targets at each iteration(')
    birth_density: State = Property(
        doc='The birth density (i.e. density from which we sample birth particles)')
    clutter_intensity: float = Property(doc='The clutter intensity per unit volume')
    num_samples: int = Property(doc='The number of samples. Default is 1024', default=1024)
    birth_scheme: str = Property(
        doc='The scheme for birth particles. Options are "expansion" | "mixture". '
            'Default is "expansion"',
        default='expansion'
    )
    start_time: datetime = Property(doc='Start time of the tracker', default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prob_detect = self.prob_detection
        self._tracks = set()

    @property
    def tracks(self):
        return self._tracks

    @property
    def prob_detect(self):
        return self._prob_detect

    @prob_detect.setter
    def prob_detect(self, prob_detect):
        if not callable(prob_detect):
            self.prob_detection = prob_detect
            self._prob_detect = self._prob_detect_simple
        else:
            self._prob_detect = copy(prob_detect)
        if hasattr(self, '_hypothesiser'):
            if hasattr(self._hypothesiser, 'hypothesiser'):
                self._hypothesiser.hypothesiser.prob_detect = self._prob_detect
            else:
                self._hypothesiser.prob_detect = self._prob_detect
        if hasattr(self, '_initiator'):
            self._initiator.filter.prob_detect = self._prob_detect

    @abstractmethod
    def track(self, detections, timestamp, *args, **kwargs):
        raise NotImplementedError

    def _prob_detect_simple(self, state_vector):
        """A simple probability of detection function."""
        return self.prob_detection

class SMCPHD_JIPDA(_BaseTracker):
    """A JIPDA tracker using an SMC-PHD filter as the track initiator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._predictor = KalmanPredictor(self.transition_model)
        self._updater = KalmanUpdater(self.measurement_model)
        self._hypothesiser = IPDAHypothesiser(self._predictor, self._updater,
                                              self.clutter_intensity,
                                              prob_detect=self.prob_detect,
                                              prob_survive=1 - self.prob_death)
        self._hypothesiser = DistanceGater(self._hypothesiser, Mahalanobis(), 10)
        self._associator = JIPDAWithEHM2(self._hypothesiser)

        resampler = SystematicResampler()
        phd_filter = ISMCPHDFilter(birth_density=self.birth_density,
                                   transition_model=self.transition_model,
                                   measurement_model=self.measurement_model,
                                   prob_detect=self.prob_detect,
                                   prob_death=self.prob_death,
                                   prob_birth=self.prob_birth,
                                   birth_rate=self.birth_rate,
                                   clutter_intensity=self.clutter_intensity,
                                   num_samples=self.num_samples,
                                   resampler=resampler,
                                   birth_scheme=self.birth_scheme)
        # Sample prior state from birth density
        if isinstance(self.birth_density, GaussianMixture):
            state_vector = np.zeros((self.transition_model.ndim_state, 0))
            particles_per_component = self.num_samples // len(self.birth_density)
            for i, component in enumerate(self.birth_density):
                if i == len(self.birth_density) - 1:
                    particles_per_component += self.num_samples % len(self.birth_density)
                particles_component = multivariate_normal.rvs(
                    component.mean.ravel(),
                    component.covar,
                    particles_per_component).T
                state_vector = np.hstack((state_vector, particles_component))
            state_vector = StateVectors(state_vector)
        else:
            state_vector = StateVectors(
                multivariate_normal.rvs(self.birth_density.state_vector.ravel(),
                                        self.birth_density.covar,
                                        size=self.num_samples).T)
        weight = np.full((self.num_samples,), Probability(1 / self.num_samples)) * self.birth_rate
        state = ParticleState(state_vector=state_vector, weight=weight, timestamp=self.start_time)

        self._initiator = ISMCPHDInitiator(filter=phd_filter, prior=state)

    def track(self, detections, timestamp, *args, **kwargs):
        tracks = list(self.tracks)
        detections = list(detections)
        num_tracks = len(tracks)
        num_detections = len(detections)

        # Perform data association
        associations = self._associator.associate(tracks, detections, timestamp)

        # Compute measurement weights
        assoc_prob_matrix = np.zeros((num_tracks, num_detections + 1))
        rho = np.zeros((len(detections)))
        for i, track in enumerate(tracks):
            for hyp in associations[track]:
                if not hyp:
                    assoc_prob_matrix[i, 0] = hyp.weight
                else:
                    j = next(d_i for d_i, detection in enumerate(detections)
                             if hyp.measurement == detection)
                    assoc_prob_matrix[i, j + 1] = hyp.weight
        for j, detection in enumerate(detections):
            # rho_tmp = 0 if len(assoc_prob_matrix) and np.sum(assoc_prob_matrix[:, j + 1]) > 0 else 1
            rho_tmp = 1
            if len(assoc_prob_matrix):
                for i, track in enumerate(tracks):
                    rho_tmp *= 1 - assoc_prob_matrix[i, j + 1]
            rho[j] = rho_tmp

        # Update tracks
        for track, multihypothesis in associations.items():

            # calculate each Track's state as a Gaussian Mixture of
            # its possible associations with each detection, then
            # reduce the Mixture to a single Gaussian State
            posterior_states = []
            posterior_state_weights = []
            for hypothesis in multihypothesis:
                posterior_state_weights.append(hypothesis.probability)
                if hypothesis:
                    posterior_states.append(self._updater.update(hypothesis))
                else:
                    posterior_states.append(hypothesis.prediction)

            # Merge/Collapse to single Gaussian
            means = StateVectors([state.state_vector for state in posterior_states])
            covars = np.stack([state.covar for state in posterior_states], axis=2)
            weights = np.asarray(posterior_state_weights)

            post_mean, post_covar = gm_reduce_single(means, covars, weights)

            track.append(GaussianStateUpdate(
                np.array(post_mean), np.array(post_covar),
                multihypothesis,
                multihypothesis[0].prediction.timestamp))

        # Initiate new tracks
        tracks = set(tracks)
        new_tracks = self._initiator.initiate(detections, timestamp, weights=rho)
        tracks |= new_tracks

        # Delete tracks that have not been updated for a while
        del_tracks = set()
        for track in tracks:
            if track.exist_prob < 0.01:
                del_tracks.add(track)
        tracks -= del_tracks

        self._tracks = set(tracks)
        return self._tracks


class SMCPHD_IGNN(_BaseTracker):
    """ A IGNN tracker using an SMC-PHD filter as the track initiator. """
    predict = Property(bool, default=True, doc="Whether to predict tracks")
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._predictor = KalmanPredictor(self.transition_model)
        self._updater = KalmanUpdater(self.measurement_model)
        self._hypothesiser = PDAHypothesiser(self._predictor, self._updater,
                                             self.clutter_intensity,
                                             prob_detect=self.prob_detect,
                                             predict=self.predict)
        self._hypothesiser = DistanceHypothesiser(self._predictor, self._updater,
                                                  Mahalanobis(), 10)
        self._associator = GNNWith2DAssignment(self._hypothesiser)

        resampler = SystematicResampler()
        phd_filter = ISMCPHDFilter(birth_density=self.birth_density,
                                   transition_model=self.transition_model,
                                   measurement_model=self.measurement_model,
                                   prob_detect=self.prob_detect,
                                   prob_death=self.prob_death,
                                   prob_birth=self.prob_birth,
                                   birth_rate=self.birth_rate,
                                   clutter_intensity=self.clutter_intensity,
                                   num_samples=self.num_samples,
                                   resampler=resampler,
                                   birth_scheme=self.birth_scheme)

        # Sample prior state from birth density
        state_vector = StateVectors(
            multivariate_normal.rvs(self.birth_density.state_vector.ravel(),
                                    self.birth_density.covar,
                                    size=self.num_samples).T)
        weight = np.full((self.num_samples,), Probability(1 / self.num_samples))
        state = ParticleState(state_vector=state_vector, weight=weight, timestamp=self.start_time)

        self._initiator = ISMCPHDInitiator(filter=phd_filter, prior=state)

    def track(self, detections, timestamp, *args, **kwargs):

        tracks = list(self.tracks)
        detections = list(detections)

        # Perform data association
        associations = self._associator.associate(tracks, detections, timestamp)

        # Compute measurement weights
        rho = np.ones((len(detections)))
        for i, track in enumerate(tracks):
            hyp = associations[track]
            if hyp:
                j = next(d_i for d_i, detection in enumerate(detections)
                         if hyp.measurement == detection)
                rho[j] = 0

        # Update tracks
        for track, hypothesis in associations.items():
            if hypothesis:
                # Update track
                state_post = self._updater.update(hypothesis)
                track.append(state_post)
                track.exist_prob = Probability(1.)
            else:
                time_interval = timestamp - track.timestamp
                track.append(hypothesis.prediction)
                prob_survive = np.exp(-self.prob_death * time_interval.total_seconds())
                track.exist_prob *= prob_survive
                non_exist_weight = 1 - track.exist_prob
                non_det_weight = (1-self.prob_detect(hypothesis.prediction)) * track.exist_prob
                track.exist_prob = non_det_weight / (non_exist_weight + non_det_weight)

        # Initiate new tracks
        tracks = set(tracks)
        new_tracks = self._initiator.initiate(detections, timestamp, weights=rho)
        tracks |= new_tracks

        # Delete tracks that have not been updated for a while
        del_tracks = set()
        for track in tracks:
            if track.exist_prob < 0.01:
                del_tracks.add(track)
        tracks -= del_tracks

        self._tracks = set(tracks)
        return self._tracks
