import copy
from typing import Sequence

import numpy as np

from .base import Predictor
from ._utils import predict_lru_cache
from .kalman import KalmanPredictor, ExtendedKalmanPredictor
from ..base import Property
from ..models.transition import TransitionModel
from ..types.prediction import Prediction
from ..types.state import GaussianState
from ..types.particle import MultiModelParticle, RaoBlackwellisedParticle


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

        return Prediction.from_state(prior, state_vector=new_state_vector, weight=prior.weight,
                                     timestamp=timestamp, particle_list=None,
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

        return Prediction.from_state(prior, state_vector=particle_prediction.state_vector,
                                     weight=particle_prediction.weight,
                                     timestamp=particle_prediction.timestamp,
                                     fixed_covar=kalman_prediction.covar, particle_list=None,
                                     transition_model=self.transition_model)


class MultiModelPredictor(Predictor):
    """MultiModelPredictor class

    An implementation of a Particle Filter predictor.
    """
    transition_model = None
    transition_models: Sequence[TransitionModel] = Property(
        doc="Transition models used to for particle transition, selected by model index on "
            "particle."
    )
    transition_matrix: np.ndarray = Property(
        doc="n-model by n-model transition matrix."
    )
    position_mapping: Sequence[Sequence[int]] = Property(
        doc="Sequence of position mappings associated with each transition model.")

    @property
    def probabilities(self):
        return np.cumsum(self.transition_matrix, axis=1)

    @predict_lru_cache()
    def predict(self, prior, control_input=None, timestamp=None, **kwargs):
        """Particle Filter prediction step

        Parameters
        ----------
        prior : :class:`~.ParticleState`
            A prior state object
        control_input : :class:`~.State`, optional
            The control input. It will only have an effect if
            :attr:`control_model` is not `None` (the default is `None`)
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
        new_particles = []
        for particle in prior.particles:
            for model_index in range(len(self.transition_matrix)):
                if particle.dynamic_model == model_index:

                    # Change the value of the dynamic value randomly according to the defined
                    # transition matrix
                    new_dynamic_model = np.searchsorted(
                        self.probabilities[model_index], np.random.random())

                    transition_model = self.transition_models[particle.dynamic_model]

                    # Based on given position mapping create a new state vector that contains only
                    # the required states
                    required_state_space = copy.copy(particle)
                    required_state_space.state_vector = particle.state_vector[
                        self.position_mapping[particle.dynamic_model]]
                    new_state_vector = transition_model.function(
                        required_state_space,
                        noise=True,
                        time_interval=time_interval,
                        **kwargs)

                    # Calculate the indices removed from the state vector to become compatible with
                    # the dynamic model
                    for j in range(len(particle.state_vector)):
                        if j not in self.position_mapping[particle.dynamic_model]:
                            new_state_vector = np.insert(new_state_vector, j, 0, axis=0)

                    new_particles.append(MultiModelParticle(new_state_vector,
                                                            weight=particle.weight,
                                                            parent=particle,
                                                            dynamic_model=new_dynamic_model))

        return Prediction.from_state(prior,
                                     state_vector=None,
                                     weight=None,
                                     particle_list=new_particles,
                                     timestamp=timestamp)


class RaoBlackwellisedMultiModelPredictor(MultiModelPredictor):
    """ParticlePredictor class

    An implementation of a Particle Filter predictor.
    """

    @predict_lru_cache()
    def predict(self, prior, control_input=None, timestamp=None, **kwargs):
        """Particle Filter prediction step

        Parameters
        ----------
        prior : :class:`~.ParticleState`
            A prior state object
        control_input : :class:`~.State`, optional
            The control input. It will only have an effect if
            :attr:`control_model` is not `None` (the default is `None`)
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

        new_particles = []
        for particle in prior.particles:

            # Change the value of the dynamic value randomly according to the defined transition
            # matrix
            new_dynamic_model = np.random.choice(
                list(range(len(particle.model_probabilities))),
                p=particle.model_probabilities)

            # Based on given position mapping create a new state vector that contains only the
            # required states
            required_state_space = particle.state_vector[
                np.array(self.position_mapping[new_dynamic_model])
            ]

            new_state_vector = self.transition_models[new_dynamic_model].function(
                required_state_space,
                noise=True,
                time_interval=time_interval,
                **kwargs)

            # Calculate the indices removed from the state vector to become compatible with the
            # dynamic model
            for j in range(len(particle.state_vector)):
                if j not in self.position_mapping[new_dynamic_model]:
                    new_state_vector = np.insert(new_state_vector, j, 0, axis=0)

            new_particles.append(
                RaoBlackwellisedParticle(new_state_vector,
                                         weight=particle.weight,
                                         parent=particle,
                                         model_probabilities=particle.model_probabilities)
            )

        return Prediction.from_state(prior,
                                     state_vector=None,
                                     weight=None,
                                     particle_list=new_particles,
                                     timestamp=timestamp)
