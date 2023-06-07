from copy import copy
from typing import List, Any, Union, Callable

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from stonesoup.base import Base, Property
from stonesoup.initiator import Initiator
from stonesoup.models.measurement import MeasurementModel
from stonesoup.models.transition import TransitionModel
from stonesoup.resampler import Resampler
from stonesoup.types.array import StateVectors
from stonesoup.types.detection import Detection, MissedDetection
from stonesoup.types.hypothesis import SingleProbabilityHypothesis
from stonesoup.types.mixture import GaussianMixture
from stonesoup.types.multihypothesis import MultipleHypothesis
from stonesoup.types.numeric import Probability
from stonesoup.types.prediction import Prediction
from stonesoup.types.state import State
from stonesoup.types.track import Track
from stonesoup.types.update import Update, GaussianStateUpdate


class SMCPHDFilter(Base):
    """
    Sequential Monte-Carlo (SMC) PHD filter implementation, based on [1]_

     .. [1] Ba-Ngu Vo, S. Singh and A. Doucet, "Sequential Monte Carlo Implementation of the
            PHD Filter for Multi-target Tracking," Sixth International Conference of Information
            Fusion, 2003. Proceedings of the, 2003, pp. 792-799, doi: 10.1109/ICIF.2003.177320.
    .. [2]  P. Horridge and S. Maskell,  “Using a probabilistic hypothesis density filter to
            confirm tracks in a multi-target environment,” in 2011 Jahrestagung der Gesellschaft
            fr Informatik, October 2011.
    """

    transition_model: TransitionModel = Property(doc='The transition model')
    measurement_model: MeasurementModel = Property(doc='The measurement model')
    prob_detect: Union[Probability, Callable[[State], Probability]] = Property(
        default=Probability(0.85),
        doc="Target Detection Probability")
    prob_death: Probability = Property(doc='The probability of death')
    prob_birth: Probability = Property(doc='The probability of birth')
    birth_rate: float = Property(
        doc='The birth rate (i.e. number of new/born targets at each iteration(')
    birth_density: State = Property(
        doc='The birth density (i.e. density from which we sample birth particles)')
    clutter_intensity: float = Property(doc='The clutter intensity per unit volume')
    resampler: Resampler = Property(default=None, doc='Resampler to prevent particle degeneracy')
    num_samples: int = Property(doc='The number of samples. Default is 1024', default=1024)
    birth_scheme: str = Property(
        doc='The scheme for birth particles. Options are "expansion" | "mixture". '
            'Default is "expansion"',
        default='expansion'
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not callable(self.prob_detect):
            prob_detect = copy(self.prob_detect)
            self.prob_detect = lambda state: prob_detect

    def predict(self, state, timestamp):
        """
        Predict the next state of the target density

        Parameters
        ----------
        state: :class:`~.State`
            The current state of the target
        timestamp: :class:`datetime.datetime`
            The time at which the state is valid

        Returns
        -------
        : :class:`~.State`
            The predicted next state of the target
        """

        prior_weights = state.weight
        time_interval = timestamp - state.timestamp

        # Predict particles forward
        pred_particles_sv = self.transition_model.function(state,
                                                           time_interval=time_interval,
                                                           noise=True)

        if self.birth_scheme == 'expansion':
            # Expansion birth scheme, as described in [1]
            # Compute number of birth particles (J_k) as a fraction of the number of particles
            num_birth = round(float(self.prob_birth) * self.num_samples)

            # Sample birth particles
            birth_particles = np.zeros((pred_particles_sv.shape[0], 0))
            if isinstance(self.birth_density, GaussianMixture):
                particles_per_component = num_birth // len(self.birth_density)
                for i, component in enumerate(self.birth_density):
                    if i == len(self.birth_density) - 1:
                        particles_per_component += num_birth % len(self.birth_density)
                    birth_particles_component = multivariate_normal.rvs(
                        component.mean.ravel(),
                        component.covar,
                        particles_per_component).T
                    birth_particles = np.hstack((birth_particles, birth_particles_component))
            else:
                birth_particles = multivariate_normal.rvs(self.birth_density.mean.ravel(),
                                                          self.birth_density.covar,
                                                          num_birth)
            birth_weights = np.full((num_birth,), Probability(self.birth_rate / num_birth))

            # Surviving particle weights
            prob_survive = np.exp(-float(self.prob_death) * time_interval.total_seconds())
            pred_weights = prob_survive * prior_weights

            # Append birth particles to predicted ones
            pred_particles_sv = StateVectors(
                np.concatenate((pred_particles_sv, birth_particles.T), axis=1))
            pred_weights = np.concatenate((pred_weights, birth_weights))
        else:
            # Mixture based birth scheme
            total_samples = self.num_samples

            # Flip a coin for each particle to decide if it gets replaced by a birth particle
            birth_inds = np.flatnonzero(np.random.binomial(1, self.prob_birth, self.num_samples))

            # Sample birth particles and replace in original state vector matrix
            num_birth = len(birth_inds)
            birth_particles = np.zeros((pred_particles_sv.shape[0], 0))
            if isinstance(self.birth_density, GaussianMixture):
                particles_per_component = num_birth // len(self.birth_density)
                for i, component in enumerate(self.birth_density):
                    if i == len(self.birth_density) - 1:
                        particles_per_component += num_birth % len(self.birth_density)
                    birth_particles_component = multivariate_normal.rvs(
                        component.mean.ravel(),
                        component.covar,
                        particles_per_component).T
                    birth_particles = np.hstack((birth_particles, birth_particles_component))
            else:
                birth_particles = multivariate_normal.rvs(self.birth_density.mean.ravel(),
                                                          self.birth_density.covar,
                                                          len(birth_inds)).T
            pred_particles_sv[:, birth_inds] = birth_particles

            # Process weights
            pred_weights = ((1 - self.prob_death) + Probability(
                self.birth_rate / total_samples)) * prior_weights

        prediction = Prediction.from_state(state, state_vector=pred_particles_sv,
                                           weight=pred_weights,
                                           timestamp=timestamp, particle_list=None,
                                           transition_model=self.transition_model)

        return prediction

    def update(self, prediction, detections, timestamp, meas_weights=None):
        """
        Update the predicted state of the target density with the given detections

        Parameters
        ----------
        prediction: :class:`~.State`
            The predicted state of the target
        detections: :class:`~.Detection`
            The detections at the current time step
        timestamp: :class:`datetime.datetime`
            The time at which the update is valid
        meas_weights: :class:`np.ndarray`
            The weights of the measurements

        Returns
        -------
        : :class:`~.State`
            The updated state of the target
        """

        weights_per_hyp = self.get_weights_per_hypothesis(prediction, detections, meas_weights)

        # Construct hypothesis objects (StoneSoup specific)
        single_hypotheses = [
            SingleProbabilityHypothesis(prediction,
                                        measurement=MissedDetection(timestamp=timestamp),
                                        probability=weights_per_hyp[:, 0])]
        for i, detection in enumerate(detections):
            single_hypotheses.append(
                SingleProbabilityHypothesis(prediction,
                                            measurement=detection,
                                            probability=weights_per_hyp[:, i + 1])
            )
        hypothesis = MultipleHypothesis(single_hypotheses, normalise=False)

        # Update weights Eq. (8) of [1]
        # w_k^i = \sum_{z \in Z_k}{w^{n,i}}, where i is the index of z in Z_k
        log_post_weights = logsumexp(np.log(weights_per_hyp).astype(float), axis=1)

        # Resample
        log_num_targets = logsumexp(log_post_weights)  # N_{k|k}
        update = copy(prediction)
        # Normalize weights
        update.weight = Probability.from_log_ufunc(log_post_weights - log_num_targets)
        if self.resampler is not None:
            update = self.resampler.resample(update, self.num_samples)  # Resample
        # De-normalize
        update.weight = Probability.from_log_ufunc(np.log(update.weight).astype(float)
                                                   + log_num_targets)

        return Update.from_state(
            state=prediction,
            state_vector=update.state_vector,
            weight=update.weight,
            particle_list=None,
            hypothesis=hypothesis,
            timestamp=timestamp)

    def iterate(self, state, detections: List[Detection], timestamp):
        """
        Iterate the filter over the given state and detections

        Parameters
        ----------
        state: :class:`~.State`
            The predicted state of the target
        detections: :class:`~.Detection`
            The detections at the current time step
        timestamp: :class:`datetime.datetime`
            The time at which the update is valid

        Returns
        -------
        : :class:`~.State`
            The updated state of the target
        """
        prediction = self.predict(state, timestamp)
        update = self.update(prediction, detections, timestamp)
        return update

    def get_measurement_loglikelihoods(self, prediction, detections, meas_weights):
        num_samples = prediction.state_vector.shape[1]
        # Compute g(z|x) matrix as in [1]
        g = np.zeros((num_samples, len(detections)))
        for i, detection in enumerate(detections):
            if not meas_weights[i]:
                g[:, i] = -np.inf
                continue
            g[:, i] = detection.measurement_model.logpdf(detection, prediction,
                                                         noise=True)
        return g

    def get_weights_per_hypothesis(self, prediction, detections, meas_weights, *args, **kwargs):
        num_samples = prediction.state_vector.shape[1]
        if meas_weights is None:
            meas_weights = np.array([Probability(1) for _ in range(len(detections))])

        # Compute g(z|x) matrix as in [1]
        g = self.get_measurement_loglikelihoods(prediction, detections, meas_weights)

        # Get probability of detection
        prob_detect = np.asfarray(self.prob_detect(prediction))

        # Calculate w^{n,i} Eq. (20) of [2]
        try:
            Ck = np.log(prob_detect[:, np.newaxis]) + g \
                 + np.log(prediction.weight[:, np.newaxis].astype(float))
        except IndexError:
            Ck = np.log(prob_detect) + g \
                 + np.log(prediction.weight[:, np.newaxis].astype(float))
        C = logsumexp(Ck, axis=0)
        k = np.log([detection.metadata['clutter_density']
                    if 'clutter_density' in detection.metadata else self.clutter_intensity
                    for detection in detections])
        C_plus = np.logaddexp(C, k)
        weights_per_hyp = np.full((num_samples, len(detections) + 1), -np.inf)
        weights_per_hyp[:, 0] = np.log(1 - prob_detect) + np.log(np.asfarray(prediction.weight))
        if len(detections):
            weights_per_hyp[:, 1:] = np.log(np.asfarray(meas_weights)) + Ck - C_plus

        return Probability.from_log_ufunc(weights_per_hyp)


class ISMCPHDFilter(SMCPHDFilter):
    def predict(self, state, timestamp):
        """
        Predict the next state of the target density

        Parameters
        ----------
        state: :class:`~.State`
            The current state of the target
        timestamp: :class:`datetime.datetime`
            The time at which the state is valid

        Returns
        -------
        : :class:`~.State`
            The predicted next state of the target
        """

        prior_weights = state.weight
        time_interval = timestamp - state.timestamp

        # Predict particles forward
        pred_particles_sv = self.transition_model.function(state,
                                                           time_interval=time_interval,
                                                           noise=True)

        # Surviving particle weights
        prob_survive = np.exp(-float(self.prob_death) * time_interval.total_seconds())
        pred_weights = prob_survive * prior_weights

        prediction = Prediction.from_state(state, state_vector=pred_particles_sv,
                                           weight=pred_weights,
                                           timestamp=timestamp, particle_list=None,
                                           transition_model=self.transition_model)
        prediction.birth_idx = state.birth_idx if hasattr(state, 'birth_idx') else []
        return prediction

    def update(self, prediction, detections, timestamp, meas_weights=None):
        """
        Update the predicted state of the target density with the given detections

        Parameters
        ----------
        prediction: :class:`~.State`
            The predicted state of the target
        detections: :class:`~.Detection`
            The detections at the current time step
        timestamp: :class:`datetime.datetime`
            The time at which the update is valid
        meas_weights: :class:`np.ndarray`
            The weights of the measurements

        Returns
        -------
        : :class:`~.State`
            The updated state of the target
        """
        num_persistent = prediction.state_vector.shape[1]
        birth_state = self.get_birth_state(prediction, detections, timestamp)

        weights_per_hyp = self.get_weights_per_hypothesis(prediction, detections, meas_weights,
                                                          birth_state)

        # Construct hypothesis objects (StoneSoup specific)
        single_hypotheses = [
            SingleProbabilityHypothesis(prediction,
                                        measurement=MissedDetection(timestamp=timestamp),
                                        probability=weights_per_hyp[:num_persistent, 0])]
        for i, detection in enumerate(detections):
            single_hypotheses.append(
                SingleProbabilityHypothesis(prediction,
                                            measurement=detection,
                                            probability=weights_per_hyp[:num_persistent,
                                                        i + 1])
            )
        hypothesis = MultipleHypothesis(single_hypotheses, normalise=False)

        # Update weights Eq. (8) of [1]
        # w_k^i = \sum_{z \in Z_k}{w^{n,i}}, where i is the index of z in Z_k
        log_post_weights = logsumexp(np.log(weights_per_hyp).astype(float), axis=1)
        log_post_weights_pers = log_post_weights[:num_persistent]
        log_post_weights_birth = log_post_weights[num_persistent:]

        # Resample persistent
        log_num_targets_pers = logsumexp(log_post_weights_pers)  # N_{k|k}
        update = copy(prediction)
        # Normalize weights
        update.weight = Probability.from_log_ufunc(log_post_weights_pers - log_num_targets_pers)
        if self.resampler is not None:
            update = self.resampler.resample(update, self.num_samples)  # Resample
        # De-normalize
        update.weight = Probability.from_log_ufunc(np.log(update.weight).astype(float)
                                                   + log_num_targets_pers)

        if len(detections):
            # Resample birth
            log_num_targets_birth = logsumexp(log_post_weights_birth)  # N_{k|k}
            update2 = copy(birth_state)
            # Normalize weights
            update2.weight = Probability.from_log_ufunc(log_post_weights_birth - log_num_targets_birth)
            if self.resampler is not None:
                update2 = self.resampler.resample(update2, update2.state_vector.shape[1])  # Resample
            # De-normalize
            update2.weight = Probability.from_log_ufunc(np.log(update2.weight).astype(float)
                                                        + log_num_targets_birth)

            full_update = Update.from_state(
                state=prediction,
                state_vector=StateVectors(np.hstack((update.state_vector, update2.state_vector))),
                weight=np.hstack((update.weight, update2.weight)),
                particle_list=None,
                hypothesis=hypothesis,
                timestamp=timestamp)
        else:
            full_update = Update.from_state(
                state=prediction,
                state_vector=update.state_vector,
                weight=update.weight,
                particle_list=None,
                hypothesis=hypothesis,
                timestamp=timestamp)
        full_update.birth_idx = [i for i in range(len(update.weight), len(full_update.weight))]
        return full_update

    def get_birth_state(self, prediction, detections, timestamp):
        # Sample birth particles
        num_birth = round(float(self.prob_birth) * self.num_samples)
        birth_particles = np.zeros((prediction.state_vector.shape[0], 0))
        birth_weights= np.zeros((0, ))
        if len(detections):
            num_birth_per_detection = num_birth // len(detections)
            for i, detection in enumerate(detections):
                if i == len(detections) - 1:
                    num_birth_per_detection += num_birth % len(detections)
                mu = self.birth_density.mean
                mu[0::2] = detection.state_vector
                cov = self.birth_density.covar
                cov[0::2, 0::2] = detection.measurement_model.covar()
                birth_particles_i = multivariate_normal.rvs(mu.ravel(),
                                                            cov,
                                                            num_birth_per_detection).T
                birth_weights_i = multivariate_normal.pdf(birth_particles_i.T,
                                                          mu.ravel(),
                                                          cov,
                                                          allow_singular=True) * Probability(self.birth_rate / num_birth)
                birth_particles = np.hstack((birth_particles, birth_particles_i))
                birth_weights = np.hstack((birth_weights, birth_weights_i))
        else:
            birth_particles = multivariate_normal.rvs(self.birth_density.mean.ravel(),
                                                      self.birth_density.covar,
                                                      num_birth).T
            birth_weights = multivariate_normal.pdf(birth_particles.T,
                                                    self.birth_density.mean.ravel(),
                                                    self.birth_density.covar,
                                                    allow_singular=True) * Probability(self.birth_rate / num_birth)
        # birth_weights = np.full((num_birth,), Probability(self.birth_rate / num_birth))
        birth_particles = StateVectors(birth_particles)
        birth_state = Prediction.from_state(prediction,
                                            state_vector=birth_particles,
                                            weight=birth_weights,
                                            timestamp=timestamp, particle_list=None,
                                            transition_model=self.transition_model)
        return birth_state

    def get_weights_per_hypothesis(self, prediction, detections, meas_weights, birth_state,
                                   *args, **kwargs):
        num_samples = prediction.state_vector.shape[1]
        if meas_weights is None:
            meas_weights = np.array([Probability(1) for _ in range(len(detections))])

        # Compute g(z|x) matrix as in [1]
        g = self.get_measurement_loglikelihoods(prediction, detections, meas_weights)

        # Get probability of detection
        prob_detect = np.asfarray(self.prob_detect(prediction))

        # Calculate w^{n,i} Eq. (20) of [2]
        try:
            Ck = np.log(prob_detect[:, np.newaxis]) + g \
                 + np.log(prediction.weight[:, np.newaxis].astype(float))
        except IndexError:
            Ck = np.log(prob_detect) + g \
                 + np.log(prediction.weight[:, np.newaxis].astype(float))
        C = logsumexp(Ck, axis=0)
        Ck_birth = np.tile(np.log(np.asfarray(birth_state.weight)[:, np.newaxis]), len(detections))
        C_birth = logsumexp(Ck_birth, axis=0)

        k = np.log([detection.metadata['clutter_density']
                    if 'clutter_density' in detection.metadata else self.clutter_intensity
                    for detection in detections])
        C_plus = np.logaddexp(C, k)
        L = np.logaddexp(C_plus, C_birth)

        weights_per_hyp = np.full((num_samples + birth_state.state_vector.shape[1],
                                   len(detections) + 1), -np.inf)
        weights_per_hyp[:num_samples, 0] = np.log(1 - prob_detect) + np.log(
            np.asfarray(prediction.weight))
        if len(detections):
            weights_per_hyp[:num_samples, 1:] = np.log(np.asfarray(meas_weights)) + Ck - L
            weights_per_hyp[num_samples:, 1:] = np.log(np.asfarray(meas_weights)) + Ck_birth - L
        return Probability.from_log_ufunc(weights_per_hyp)


class SMCPHDInitiator(Initiator):
    filter: SMCPHDFilter = Property(doc='The phd filter')
    prior: Any = Property(doc='The prior state')
    threshold: Probability = Property(doc='The thrshold probability for initiation',
                                      default=Probability(0.9))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._state = self.prior

    def initiate(self, detections, timestamp, weights=None, **kwargs):
        tracks = set()

        if self._state.timestamp is None:
            self._state.timestamp = timestamp
        # Predict forward
        prediction = self.filter.predict(self._state, timestamp)

        # Calculate weights per hypothesis
        weights_per_hyp = self.filter.get_weights_per_hypothesis(prediction, detections, weights)
        log_weights_per_hyp = np.log(weights_per_hyp).astype(float)

        # Calculate intensity per hypothesis
        log_intensity_per_hyp = logsumexp(log_weights_per_hyp, axis=0)

        # Find detections with intensity above threshold and initiate
        valid_inds = np.flatnonzero(np.exp(log_intensity_per_hyp) > self.threshold)
        for idx in valid_inds:
            if not idx:
                continue

            particles_sv = copy(prediction.state_vector)
            weight = np.exp(log_weights_per_hyp[:, idx] - log_intensity_per_hyp[idx])

            mu = np.average(particles_sv,
                            axis=1,
                            weights=weight)
            cov = np.cov(particles_sv, ddof=0, aweights=weight)

            hypothesis = SingleProbabilityHypothesis(prediction,
                                                     measurement=detections[idx-1],
                                                     probability=weights_per_hyp[:, idx])

            track_state = GaussianStateUpdate(mu, cov, hypothesis=hypothesis,
                                              timestamp=timestamp)

            # if np.trace(track_state.covar) < 10:
            weights_per_hyp[:, idx] = Probability(0)
            track = Track([track_state])
            track.exist_prob = Probability(log_intensity_per_hyp[idx], log_value=True)
            tracks.add(track)

            weights[idx-1] = 0

        # Update filter
        self._state = self.filter.update(prediction, detections, timestamp, weights)

        return tracks


class ISMCPHDInitiator(SMCPHDInitiator):
    filter: ISMCPHDFilter = Property(doc='The phd filter')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._state = self.prior

    def initiate(self, detections, timestamp, weights=None, **kwargs):
        tracks = set()

        if self._state.timestamp is None:
            self._state.timestamp = timestamp
        # Predict forward
        prediction = self.filter.predict(self._state, timestamp)

        # Calculate weights per hypothesis
        birth_state = self.filter.get_birth_state(prediction, detections, timestamp)
        weights_per_hyp = self.filter.get_weights_per_hypothesis(prediction, detections, weights, birth_state)
        log_weights_per_hyp = np.log(weights_per_hyp[:self.filter.num_samples, :]).astype(float)

        # Calculate intensity per hypothesis
        log_intensity_per_hyp = logsumexp(log_weights_per_hyp, axis=0)
        # print(np.exp(log_intensity_per_hyp))
        # Find detections with intensity above threshold and initiate
        valid_inds = np.flatnonzero(np.exp(log_intensity_per_hyp) > self.threshold)
        for idx in valid_inds:
            if not idx:
                continue

            particles_sv = copy(prediction.state_vector[:, :len(prediction)-len(prediction.birth_idx)])
            weight = np.exp(log_weights_per_hyp[:self.filter.num_samples, idx] - log_intensity_per_hyp[idx])

            mu = np.average(particles_sv,
                            axis=1,
                            weights=weight)
            cov = np.cov(particles_sv, ddof=0, aweights=weight)

            hypothesis = SingleProbabilityHypothesis(prediction,
                                                     measurement=detections[idx - 1],
                                                     probability=weights_per_hyp[:self.filter.num_samples, idx])

            track_state = GaussianStateUpdate(mu, cov, hypothesis=hypothesis,
                                              timestamp=timestamp)

            # if np.trace(track_state.covar) < 10:
            weights_per_hyp[:, idx] = Probability(0)
            track = Track([track_state])
            track.exist_prob = Probability(log_intensity_per_hyp[idx], log_value=True)
            tracks.add(track)

            weights[idx - 1] = 0

        # Update filter
        self._state = self.filter.update(prediction, detections, timestamp, weights)

        return tracks