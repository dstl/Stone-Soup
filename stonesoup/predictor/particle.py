import copy
from typing import Sequence

import numpy as np
from scipy.special import logsumexp
from ordered_set import OrderedSet

from .base import Predictor
from ._utils import predict_lru_cache
from .kalman import KalmanPredictor, ExtendedKalmanPredictor
from ..base import Property
from ..models.transition import TransitionModel
from ..types.prediction import Prediction
from ..types.state import GaussianState
from ..sampler import Sampler

from ..types.array import StateVectors
from ..types.numeric import Probability


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
                                     state_vector=new_state_vector,
                                     timestamp=timestamp,
                                     transition_model=self.transition_model)


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
                                     transition_model=self.transition_model)


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
        new_weight_vector = np.concatenate(
            (new_particle_state.weight, np.array([Probability(1/nbirth_particles)]
                                                 * nbirth_particles)))
        new_particle_state.weight = new_weight_vector/sum(new_weight_vector)

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

        predicted_weights = self.predict_weights(new_particle_state.existence_probability,
                                                 predicted_existence, nsurv_particles,
                                                 new_particle_state.weight, nbirth_particles)
        new_particle_state.weight = predicted_weights

        # Create the prediction output
        new_particle_state = Prediction.from_state(
            new_particle_state,
            state_vector=new_state_vector,
            existence_probability=predicted_existence,
            parent=untransitioned_state,
            particle_list=None,
            timestamp=timestamp,
        )
        return new_particle_state

    def estimate_existence(self, existence_prior):
        existence_estimate = self.birth_probability * (1 - existence_prior) \
                             + self.survival_probability * existence_prior
        return existence_estimate

    def predict_weights(self, prior_existence, predicted_existence, surv_part_size, prior_weights,
                        nbirth_particles):

        # Weight prediction function currently assumes that the chosen importance density is the
        # transition density. This will need to change if implementing a different importance
        # density or incorporating visibility information

        surv_weights = (self.survival_probability*prior_existence)/predicted_existence \
            * prior_weights[:surv_part_size]

        birth_weights = (self.birth_probability*(1-prior_existence))/predicted_existence \
            * prior_weights[nbirth_particles:]
        predicted_weights = np.concatenate((surv_weights, birth_weights))

        # Normalise weights
        predicted_weights = predicted_weights / np.sum(predicted_weights)

        return predicted_weights

    @staticmethod
    def get_detections(prior):

        detections = OrderedSet()
        if hasattr(prior, 'hypothesis'):
            if prior.hypothesis is not None:
                for hypothesis in prior.hypothesis:
                    detections |= {hypothesis.measurement}

        return detections
