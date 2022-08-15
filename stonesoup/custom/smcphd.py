import copy
from typing import List

import numpy as np
from scipy.stats import multivariate_normal

from stonesoup.base import Base, Property
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
from stonesoup.types.update import Update


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

    def iterate(self, state, detections: List[Detection], timestamp):
        prior_weights = state.weight
        time_interval = timestamp - state.timestamp
        detections_list = list(detections)

        # 1) Predict
        # =======================================================================================>

        # Predict particles forward
        pred_particles_sv = self.transition_model.function(state,
                                                           time_interval=time_interval,
                                                           noise=True)

        if self.birth_scheme == 'expansion':
            # Expansion birth scheme, as described in [1]
            # Compute number of birth particles (J_k) as a fraction of the number of particles
            num_birth = round(float(self.prob_birth * self.num_samples))

            total_samples = self.num_samples + num_birth    # L_{k-1} + Jk

            # Sample birth particles
            birth_particles = multivariate_normal.rvs(self.birth_density.mean.ravel(),
                                                      self.birth_density.covar,
                                                      num_birth)
            birth_weights = np.ones((num_birth,)) * Probability(self.birth_rate / num_birth)

            # Surviving particle weights
            pred_weights = (1 - self.prob_death) * prior_weights

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

        # 2) Update
        # =======================================================================================>

        # Compute g(z|x) likelihood matrix as in [1]
        g = np.zeros((total_samples, len(detections)), dtype=Probability)
        for i, detection in enumerate(detections_list):
            g[:, i] = detection.measurement_model.pdf(detection, prediction,
                                                      noise=True)

        # Calculate w^{n,i} Eq. (20) of [2]
        # (i.e. the individual sum terms inside the square brackets in Eq. (8) of [1], multiplied
        #  by the corresponding predicted weight w_{k|k-1}^(i))
        weights_per_hyp = np.zeros((total_samples, len(detections) + 1), dtype=Probability)
        weights_per_hyp[:, 0] = (1 - self.prob_detect) * pred_weights   # Null hypothesis
        if len(detections):
            # C = \psi_{k,z}(x_k^(i)) * w_{k|k-1}^(i) in Eq. (8) of [1]
            C = self.prob_detect * g * pred_weights[:, np.newaxis]
            Ck = np.sum(C, axis=0)  # C_k(z) (Eq. (9) of [1])
            C_plus = Ck + self.clutter_intensity  # \kappa_{k}(z) + Ck(z) term in Eq. (8) of [1]
            weights_per_hyp[:, 1:] = C / C_plus  # True-detection hypotheses

        # Construct hypothesis objects (StoneSoup specific)
        intensity_per_hyp = np.sum(weights_per_hyp, axis=0)
        single_hypotheses = [
            SingleProbabilityHypothesis(prediction,
                                        measurement=MissedDetection(timestamp=timestamp),
                                        probability=Probability(intensity_per_hyp[0]))]
        for i, detection in enumerate(detections_list):
            single_hypotheses.append(
                SingleProbabilityHypothesis(prediction,
                                            measurement=detection,
                                            probability=Probability(intensity_per_hyp[i + 1]))
            )
        hypothesis = MultipleHypothesis(single_hypotheses, normalise=False)

        # Update weights Eq. (8) of [1]
        # w_k^i = \sum_{z \in Z_k}{w^{n,i}}, where i is the index of z in Z_k
        post_weights = np.sum(weights_per_hyp, axis=1)

        # Resample
        num_targets = np.sum(post_weights)  # N_{k|k}
        update = copy.copy(prediction)
        update.weight = post_weights / num_targets  # Normalize weights
        if self.resampler is not None:
            update = self.resampler.resample(update, self.num_samples)  # Resample
        update.weight = np.array(update.weight) * num_targets   # De-normalize

        return Update.from_state(
            state=prediction,
            state_vector=update.state_vector,
            weight=update.weight,
            particle_list=None,
            hypothesis=hypothesis,
            timestamp=timestamp)