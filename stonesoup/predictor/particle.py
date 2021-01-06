# -*- coding: utf-8 -*-
from copy import copy

from .base import Predictor
from ._utils import predict_lru_cache
from ..types.particle import Particle
from ..types.prediction import Prediction


class ParticlePredictor(Predictor):
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
            new_particle = copy(particle)
            new_state_vector = self.transition_model.function(
                particle,
                noise=True,
                time_interval=time_interval,
                **kwargs)
            new_particle.state_vector = new_state_vector
            new_particles.append(new_particle)

        return Prediction.from_state(prior, particles=new_particles, timestamp=timestamp)
