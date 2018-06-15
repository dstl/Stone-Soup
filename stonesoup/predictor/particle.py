from functools import lru_cache
from .base import Predictor
from ..types.particle import Particle, ParticleState

class ParticlePredictor(Predictor):
    """ParticlePredictor class

    An implementation of a Particle Filter predictor.

    """

    def predict(self, state, control_input=None, timestamp=None, **kwargs):
        """Particle Filter full prediction step

        Parameters
        ----------
        state : :class:`stonesoup.types.state.ParticleState`
            A prior state object
        control_input : :class:`stonesoup.types.state.State`, optional
            The control input. It will only have an effect if
            :attr:`control_model` is not `None` (the default is `None`)
        timestamp: :class:`datetime.datetime`, optional
            A timestamp signifying when the prediction is performed \
            (the default is `None`)

        Returns
        -------
        :class:`stonesoup.types.state.ParticleState`
            The predicted state
        """

        state_pred = self.predict_state(state,control_input,
                                             timestamp, **kwargs)
        return state_pred

    @lru_cache()
    def predict_state(self, state, control_input=None,
                      timestamp=None, **kwargs):
        """Particle Filter state prediction step

        Parameters
        ----------
        state : :class:`stonesoup.types.state.ParticleState`
            The prior state
        control_input : :class:`stonesoup.types.state.State`, optional
            The control input. It will only have an effect if
            :attr:`control_model` is not `None` (the default is `None`)
        timestamp: :class:`datetime.datetime`, optional
            A timestamp signifying when the prediction is performed \
            (the default is `None`)

        Returns
        -------
        :class:`stonesoup.types.state.ParticleState`
            The predicted state

        """

        # Compute time_interval
        try:
            time_interval = timestamp - state.timestamp
        except TypeError as e:
            # TypeError: (timestamp or state.timestamp) is None
            time_interval = None

        new_particles = []
        for particle in state.particles:
            new_state_vector = self.transition_model.function(particle.state_vector, time_interval = time_interval, **kwargs)
            new_particles.append(Particle(new_state_vector, weight=particle.weight, timestamp=timestamp, parent = particle.parent))

        return ParticleState(new_particles,timestamp)
