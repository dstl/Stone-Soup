from copy import copy
from typing import List, Any

import numpy as np
from scipy.stats import multivariate_normal

from stonesoup.base import Base, Property
from stonesoup.initiator import Initiator
from stonesoup.models.measurement import MeasurementModel
from stonesoup.models.transition import TransitionModel
from stonesoup.resampler import Resampler
from stonesoup.types.array import StateVectors
from stonesoup.types.detection import Detection, MissedDetection
from stonesoup.types.hypothesis import SingleProbabilityHypothesis
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
    prob_detect: Probability = Property(doc='The probability of detection')
    prob_death: Probability = Property(doc='The probability of death')
    prob_birth: Probability = Property(doc='The probability of birth')
    birth_rate: float = Property(doc='The birth rate (i.e. number of new/born targets at each iteration(')
    birth_density: State = Property(doc='The birth density (i.e. density from which we sample birth particles)')
    clutter_intensity: float = Property(doc='The clutter intensity per unit volume')
    resampler: Resampler = Property(default=None, doc='Resampler to prevent particle degeneracy')
    num_samples: int = Property(doc='The number of samples. Default is 1024', default=1024)
    birth_scheme: str = Property(
        doc='The scheme for birth particles. Options are "expansion" | "mixture". '
            'Default is "expansion"',
        default='expansion'
    )

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
            num_birth = round(float(self.prob_birth * self.num_samples))

            # Sample birth particles
            birth_particles = multivariate_normal.rvs(self.birth_density.mean.ravel(),
                                                      self.birth_density.covar,
                                                      num_birth)
            birth_weights = np.ones((num_birth,)) * Probability(self.birth_rate / num_birth)

            # Surviving particle weights
            prob_survive = np.exp(-self.prob_death*time_interval.total_seconds())
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
            birth_particles = multivariate_normal.rvs(self.birth_density.mean.ravel(),
                                                      self.birth_density.covar,
                                                      len(birth_inds))
            pred_particles_sv[:, birth_inds] = birth_particles.T

            # Process weights
            pred_weights = ((1 - self.prob_death) + Probability(self.birth_rate / total_samples)) * prior_weights

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
        post_weights = np.sum(weights_per_hyp, axis=1)

        # Resample
        num_targets = np.sum(post_weights)  # N_{k|k}
        update = copy(prediction)
        update.weight = post_weights / num_targets  # Normalize weights
        if self.resampler is not None:
            update = self.resampler.resample(update, self.num_samples)  # Resample
        update.weight = np.array(update.weight) * num_targets  # De-normalize

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

    def get_measurement_likelihoods(self, prediction, detections, meas_weights):
        num_samples = prediction.state_vector.shape[1]
        # Compute g(z|x) matrix as in [1]
        g = np.zeros((num_samples, len(detections)), dtype=Probability)
        for i, detection in enumerate(detections):
            if not meas_weights[i]:
                g[:, i] = Probability(0)
                continue
            g[:, i] = detection.measurement_model.pdf(detection, prediction,
                                                      noise=True)
        return g

    def get_weights_per_hypothesis(self, prediction, detections, meas_weights):
        num_samples = prediction.state_vector.shape[1]
        if meas_weights is None:
            meas_weights = np.array([Probability(1) for _ in range(len(detections))])

        # Compute g(z|x) matrix as in [1]
        g = self.get_measurement_likelihoods(prediction, detections, meas_weights)

        # Calculate w^{n,i} Eq. (20) of [2]
        Ck = meas_weights * self.prob_detect * g * prediction.weight[:, np.newaxis]
        C = np.sum(Ck, axis=0)
        k = np.array([detection.metadata['clutter_density']
                      if 'clutter_density' in detection.metadata else self.clutter_intensity
                      for detection in detections])
        C_plus = C + k

        weights_per_hyp = np.zeros((num_samples, len(detections) + 1), dtype=Probability)
        weights_per_hyp[:, 0] = (1 - self.prob_detect) * prediction.weight
        if len(detections):
            weights_per_hyp[:, 1:] = Ck / C_plus

        return weights_per_hyp


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

        # Predict forward
        prediction = self.filter.predict(self._state, timestamp)

        weights_per_hyp = self.filter.get_weights_per_hypothesis(prediction, detections, weights)

        intensity_per_hyp = np.sum(weights_per_hyp, axis=0)
        valid_inds = np.flatnonzero(intensity_per_hyp > self.threshold)
        for idx in valid_inds:
            if not idx:
                continue

            particles_sv = copy(prediction.state_vector)
            weight = weights_per_hyp[:, idx] / intensity_per_hyp[idx]

            mu = np.average(particles_sv,
                            axis=1,
                            weights=weight)
            cov = np.cov(particles_sv, ddof=0, aweights=np.array(weight))

            hypothesis = SingleProbabilityHypothesis(prediction,
                                                     measurement=detections[idx-1],
                                                     probability=weights_per_hyp[:, idx])

            track_state = GaussianStateUpdate(mu, cov, hypothesis=hypothesis,
                                              timestamp=timestamp)

            # if np.trace(track_state.covar) < 10:
            weights_per_hyp[:, idx] = Probability(0)
            track = Track([track_state])
            track.exist_prob = intensity_per_hyp[idx]
            tracks.add(track)

            weights[idx-1] = 0

        self._state = self.filter.update(prediction, detections, timestamp, weights)
        return tracks
