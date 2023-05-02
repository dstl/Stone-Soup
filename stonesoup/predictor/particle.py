import copy
from scipy.stats import multivariate_normal
from typing import Sequence, Union, Literal

import numpy as np
from scipy.special import logsumexp
from ordered_set import OrderedSet

from .base import Predictor
from ._utils import predict_lru_cache
from .kalman import KalmanPredictor, ExtendedKalmanPredictor
from ..base import Property
from ..models.transition import TransitionModel
from ..types.mixture import GaussianMixture
from ..types.numeric import Probability
from ..types.prediction import Prediction
from ..types.state import GaussianState
from ..sampler import Sampler

from ..types.array import StateVectors


class ParticlePredictor(Predictor):
    """ParticlePredictor class

    An implementation of a Particle Filter predictor.
    """

    @predict_lru_cache()
    def predict(self, prior, timestamp=None, **kwargs):
        """Particle Filter prediction step

        Parameters
        ----------
        prior : :class:`~.ParticleState`
            A prior state object
        timestamp: :class:`datetime.datetime`, optional
            A timestamp signifying when the prediction is performed
            (the default is `None`)

        Returns
        -------
        : :class:`~.ParticleStatePrediction`
            The predicted state
        """
        # Compute time_interval
        try:
            time_interval = timestamp - prior.timestamp
        except TypeError:
            # TypeError: (timestamp or prior.timestamp) is None
            time_interval = None

        new_state_vector = self.transition_model.function(
            prior,
            noise=True,
            time_interval=time_interval,
            **kwargs)

        return Prediction.from_state(prior,
                                     parent=prior,
                                     state_vector=new_state_vector,
                                     timestamp=timestamp,
                                     transition_model=self.transition_model,
                                     prior=prior)


class ParticleFlowKalmanPredictor(ParticlePredictor):
    """Gromov Flow Parallel Kalman Particle Predictor

    This is a wrapper around the :class:`~.GromovFlowParticlePredictor` which
    can use a :class:`~.ExtendedKalmanPredictor` or
    :class:`~.UnscentedKalmanPredictor` in parallel in order to maintain a
    state covariance, as proposed in [1]_.

    This should be used in conjunction with the
    :class:`~.ParticleFlowKalmanUpdater`.

    Parameters
    ----------

    References
    ----------
    .. [1] Ding, Tao & Coates, Mark J., "Implementation of the Daum-Huang
       Exact-Flow Particle Filter" 2012
    """
    kalman_predictor: KalmanPredictor = Property(
        default=None,
        doc="Kalman predictor to use. Default `None` where a new instance of"
            ":class:`~.ExtendedKalmanPredictor` will be created utilising the"
            "same transition model.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.kalman_predictor is None:
            self.kalman_predictor = ExtendedKalmanPredictor(
                self.transition_model)

    def predict(self, prior, *args, **kwargs):
        particle_prediction = super().predict(prior, *args, **kwargs)

        kalman_prediction = self.kalman_predictor.predict(
            GaussianState(prior.state_vector, prior.covar, prior.timestamp),
            *args, **kwargs)

        return Prediction.from_state(prior,
                                     state_vector=particle_prediction.state_vector,
                                     log_weight=particle_prediction.log_weight,
                                     timestamp=particle_prediction.timestamp,
                                     fixed_covar=kalman_prediction.covar,
                                     transition_model=self.transition_model,
                                     prior=prior)


class MultiModelPredictor(Predictor):
    """MultiModelPredictor class

    An implementation of a Particle Filter predictor utilising multiple models.
    """
    transition_model = None
    transition_models: Sequence[TransitionModel] = Property(
        doc="Transition models used to for particle transition, selected by model index on "
            "particle. Models dimensions can be subset of the overall state space, by "
            "using :attr:`model_mappings`."
    )
    transition_matrix: np.ndarray = Property(
        doc="n-model by n-model transition matrix."
    )
    model_mappings: Sequence[Sequence[int]] = Property(
        doc="Sequence of mappings associated with each transition model. This enables mapping "
            "between model and state space, enabling use of models that may have different "
            "dimensions (e.g. velocity or acceleration). Parts of the state that aren't mapped "
            "are set to zero.")

    @property
    def probabilities(self):
        return np.cumsum(self.transition_matrix, axis=1)

    @predict_lru_cache()
    def predict(self, prior, timestamp=None, **kwargs):
        """Particle Filter prediction step

        Parameters
        ----------
        prior : :class:`~.MultiModelParticleState`
            A prior state object
        timestamp: :class:`datetime.datetime`, optional
            A timestamp signifying when the prediction is performed
            (the default is `None`)
        Returns
        -------
        : :class:`~.ParticleStatePrediction`
            The predicted state
        """

        new_particle_state = Prediction.from_state(
            prior,
            state_vector=copy.copy(prior.state_vector),
            parent=prior,
            dynamic_model=copy.copy(prior.dynamic_model),
            timestamp=timestamp)

        for model_index, transition_model in enumerate(self.transition_models):
            # Change the value of the dynamic value randomly according to the defined
            # transition matrix
            current_model_indices = prior.dynamic_model == model_index
            current_model_count = np.count_nonzero(current_model_indices)
            if current_model_count == 0:
                continue
            new_dynamic_models = np.searchsorted(
                self.probabilities[model_index],
                np.random.random(size=current_model_count))

            new_state_vector = self.apply_model(
                prior[current_model_indices], transition_model, timestamp,
                self.model_mappings[model_index], noise=True, **kwargs)

            new_particle_state.state_vector[:, current_model_indices] = new_state_vector
            new_particle_state.dynamic_model[current_model_indices] = new_dynamic_models

        return new_particle_state

    @staticmethod
    def apply_model(prior, transition_model, timestamp, model_mapping, **kwargs):
        # Based on given position mapping create a new state vector that contains only the
        # required states
        orig_ndim = prior.ndim
        prior.state_vector = prior.state_vector[model_mapping, :]
        new_state_vector = transition_model.function(
            prior, time_interval=timestamp - prior.timestamp, **kwargs)

        # Calculate the indices removed from the state vector to become compatible with the
        # dynamic model
        for j in range(orig_ndim):
            if j not in model_mapping:
                new_state_vector = np.insert(new_state_vector, j, 0, axis=0)

        return new_state_vector


class RaoBlackwellisedMultiModelPredictor(MultiModelPredictor):
    """Rao-Blackwellised Multi Model Predictor class

    An implementation of a Particle Filter predictor utilising multiple models, with per
    particle model probabilities.
    """

    @predict_lru_cache()
    def predict(self, prior, timestamp=None, **kwargs):
        """Particle Filter prediction step

        Parameters
        ----------
        prior : :class:`~.RaoBlackwellisedParticleState`
            A prior state object
        timestamp: :class:`datetime.datetime`, optional
            A timestamp signifying when the prediction is performed
            (the default is `None`)
        Returns
        -------
        : :class:`~.ParticleStatePrediction`
            The predicted state
        """

        new_particle_state = Prediction.from_state(
            prior,
            state_vector=copy.copy(prior.state_vector),
            parent=prior,
            timestamp=timestamp)

        # Change the value of the dynamic value randomly according to the
        # matrix
        new_dynamic_models = np.argmax(
            prior.model_probabilities.astype(float).cumsum(0) > np.random.rand(len(prior)),
            axis=0)

        for model_index, transition_model in enumerate(self.transition_models):

            new_model_indices = new_dynamic_models == model_index
            if np.count_nonzero(new_model_indices) == 0:
                continue

            new_state_vector = self.apply_model(
                prior[new_model_indices], transition_model, timestamp,
                self.model_mappings[model_index], noise=True, **kwargs)

            new_particle_state.state_vector[:, new_model_indices] = new_state_vector

        return new_particle_state


class BernoulliParticlePredictor(ParticlePredictor):
    """Bernoulli Particle Filter Predictor class

    An implementation of a particle filter predictor utilising the Bernoulli
    filter formulation that estimates the spatial distribution of a single
    target and estimates its existence, as described in [2]_.

    This should be used in conjunction with the
    :class:`~.BernoulliParticleUpdater`.

    References
    ----------
    .. [2] Ristic, Branko & Vo, Ba-Toung & Vo, Ba-Ngu & Farina, Alfonso, A
       Tutorial on Bernoulli Filters: Theory, Implementation and Applications,
       2013, IEEE Transactions on Signal Processing, 61(13), 3406-3430.
    """

    birth_probability: float = Property(
        default=0.01,
        doc="Probability of target birth.")
    survival_probability: float = Property(
        default=0.98,
        doc="Probability of target survival")
    birth_sampler: Sampler = Property(
        default=None,
        doc="Sampler object used for sampling birth particles. "
            "Currently implementation assumes the :class:`~.DetectionSampler` is used")

    def predict(self, prior, timestamp=None, **kwargs):
        """Bernoulli Particle Filter prediction step

        Parameters
        ----------
        prior : :class:`~.BernoulliParticleState`
            A prior state object
        timestamp : :class:`~.datetime.datetime`
            A timestamp signifying when the prediction is performed
            (the default is `None`)
        Returns
        -------
        : :class:`~.ParticleStatePrediction`
            The predicted state and existence"""

        new_particle_state = copy.copy(prior)

        nsurv_particles = new_particle_state.state_vector.shape[1]

        # Calculate time interval
        try:
            time_interval = timestamp - new_particle_state.timestamp
        except TypeError:
            # TypeError: (timestamp or prior.timestamp) is None
            time_interval = None

        # Sample from birth distribution
        detections = self.get_detections(prior)
        birth_state = self.birth_sampler.sample(detections)

        birth_part = birth_state.state_vector
        nbirth_particles = len(birth_state)

        # Unite the surviving and birth particle sets in the prior
        new_particle_state.state_vector = StateVectors(np.concatenate(
            (new_particle_state.state_vector, birth_part), axis=1))
        # Extend weights to match length of state_vector
        new_log_weight_vector = np.concatenate(
            (new_particle_state.log_weight, np.full(nbirth_particles, np.log(1/nbirth_particles))))
        new_particle_state.log_weight = new_log_weight_vector - logsumexp(new_log_weight_vector)

        untransitioned_state = Prediction.from_state(
            new_particle_state,
            parent=prior,
        )

        # Predict particles using the transition model
        new_state_vector = self.transition_model.function(
            new_particle_state,
            noise=True,
            time_interval=time_interval,
            **kwargs)

        # Estimate existence
        predicted_existence = self.estimate_existence(
            new_particle_state.existence_probability)

        predicted_log_weights = self.predict_log_weights(
            new_particle_state.existence_probability,
            predicted_existence, nsurv_particles,
            new_particle_state.log_weight)

        # Create the prediction output
        new_particle_state = Prediction.from_state(
            new_particle_state,
            state_vector=new_state_vector,
            log_weight=predicted_log_weights,
            existence_probability=predicted_existence,
            parent=untransitioned_state,
            timestamp=timestamp,
            transition_model=self.transition_model,
            prior=prior
        )
        return new_particle_state

    def estimate_existence(self, existence_prior):
        existence_estimate = self.birth_probability * (1 - existence_prior) \
                             + self.survival_probability * existence_prior
        return existence_estimate

    def predict_log_weights(self, prior_existence, predicted_existence, surv_part_size,
                            prior_log_weights):

        # Weight prediction function currently assumes that the chosen importance density is the
        # transition density. This will need to change if implementing a different importance
        # density or incorporating visibility information

        surv_weights = np.log((self.survival_probability*prior_existence)/predicted_existence) \
            + prior_log_weights[:surv_part_size]

        birth_weights = np.log((self.birth_probability*(1-prior_existence))/predicted_existence) \
            + prior_log_weights[surv_part_size:]
        predicted_log_weights = np.concatenate((surv_weights, birth_weights))

        # Normalise weights
        predicted_log_weights -= logsumexp(predicted_log_weights)

        return predicted_log_weights

    @staticmethod
    def get_detections(prior):

        detections = OrderedSet()
        if hasattr(prior, 'hypothesis'):
            if prior.hypothesis is not None:
                for hypothesis in prior.hypothesis:
                    detections |= {hypothesis.measurement}

        return detections


class SMCPHDPredictor(Predictor):
    """SMC-PHD Predictor class

    Sequential Monte-Carlo (SMC) PHD predictor implementation, based on [1]_.

    Notes
    -----
    - It is assumed that the proposal distribution is the same as the dynamics
    - Target "spawing" is not implemented

     .. [1] Ba-Ngu Vo, S. Singh and A. Doucet, "Sequential Monte Carlo Implementation of the
            PHD Filter for Multi-target Tracking," Sixth International Conference of Information
            Fusion, 2003. Proceedings of the, 2003, pp. 792-799, doi: 10.1109/ICIF.2003.177320.
    """
    prob_death: Probability = Property(
        doc="The probability of death")
    prob_birth: Probability = Property(
        doc="The probability of birth")
    birth_rate: float = Property(
        doc="The birth rate (i.e. number of new/born targets at each iteration)")
    birth_density: Union[GaussianState, GaussianMixture] = Property(
        doc="The birth density (i.e. density from which to sample birth particles)")
    birth_scheme: Literal["expansion", "mixture"] = Property(
        default="expansion",
        doc="The scheme for birth particles. Options are 'expansion' | 'mixture'. "
            "Default is 'expansion'"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.birth_scheme not in ["expansion", "mixture"]:
            raise ValueError("Invalid birth scheme. Options are 'expansion' | 'mixture'")

    @predict_lru_cache()
    def predict(self, prior, timestamp=None, **kwargs):
        """ SMC-PHD prediction step

        Parameters
        ----------
        prior: :class:`~.ParticleState`
            The prior state
        timestamp: :class:`datetime.datetime`
            The time at which to predict the next state

        Returns
        -------
        : :class:`~.ParticleStatePrediction`
            The predicted state

        """
        num_samples = len(prior)
        log_prior_weights = prior.log_weight
        time_interval = timestamp - prior.timestamp

        # Predict surviving particles forward
        pred_particles_sv = self.transition_model.function(prior,
                                                           time_interval=time_interval,
                                                           noise=True)

        # Perform birth and update weights
        num_state_dim = pred_particles_sv.shape[0]
        if self.birth_scheme == "expansion":
            # Expansion birth scheme, as described in [1]
            # Compute number of birth particles (J_k) as a fraction of the number of particles
            num_birth = round(float(self.prob_birth) * num_samples)

            # Sample birth particles
            birth_particles_sv = self._sample_birth_particles(num_state_dim, num_birth)
            log_birth_weights = np.full((num_birth,), np.log(self.birth_rate / num_birth))

            # Surviving particle weights
            log_prob_survive = -float(self.prob_death) * time_interval.total_seconds()
            log_pred_weights = log_prob_survive + log_prior_weights

            # Append birth particles to predicted ones
            pred_particles_sv = StateVectors(
                np.concatenate((pred_particles_sv, birth_particles_sv), axis=1))
            log_pred_weights = np.concatenate((log_pred_weights, log_birth_weights))
        else:
            # Flip a coin for each particle to decide if it gets replaced by a birth particle
            birth_inds = np.flatnonzero(np.random.binomial(1, float(self.prob_birth), num_samples))

            # Sample birth particles and replace in original state vector matrix
            num_birth = len(birth_inds)
            birth_particles_sv = self._sample_birth_particles(num_state_dim, num_birth)
            pred_particles_sv[:, birth_inds] = birth_particles_sv

            # Process weights
            prob_survive = np.exp(-float(self.prob_death) * time_interval.total_seconds())
            birth_weight = self.birth_rate / num_samples
            log_pred_weights = np.log(prob_survive + birth_weight) + log_prior_weights

        prediction = Prediction.from_state(prior, state_vector=pred_particles_sv,
                                           log_weight=log_pred_weights,
                                           timestamp=timestamp, particle_list=None,
                                           transition_model=self.transition_model)

        return prediction

    def _sample_birth_particles(self, num_state_dim: int, num_birth: int):
        birth_particles = np.zeros((num_state_dim, 0))
        if isinstance(self.birth_density, GaussianMixture):
            n_parts_per_component = num_birth // len(self.birth_density)
            for i, component in enumerate(self.birth_density):
                if i == len(self.birth_density) - 1:
                    n_parts_per_component += num_birth % len(self.birth_density)
                birth_particles_component = multivariate_normal.rvs(
                    component.mean.ravel(),
                    component.covar,
                    n_parts_per_component).T
                birth_particles = np.hstack((birth_particles, birth_particles_component))
        else:
            birth_particles = np.atleast_2d(multivariate_normal.rvs(
                self.birth_density.mean.ravel(),
                self.birth_density.covar,
                num_birth)).T
        return birth_particles
