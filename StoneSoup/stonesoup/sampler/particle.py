import numpy as np

from .base import Sampler
from ..base import Property
from typing import Callable
from ..types.state import ParticleState
from ..types.array import StateVectors


class ParticleSampler(Sampler):
    """Particle sampler.

     A generic :class:`~.Sampler` which wraps around most distribution sampling functions from
     :class:`numpy` and :class:`scipy`, that returns a :class:`~.ParticleState`
     """

    distribution_func: Callable = Property(
        doc="Callable function that returns samples from the desired distribution.")

    params: dict = Property(
        doc="Dictionary containing the keyword arguments for :attr:`distribution_func`.")

    ndim_state: int = Property(
        doc="Number of dimensions in each sample.")

    def sample(self, params=None, timestamp=None):
        """Samples from the desired distribution and returns as a :class:`~.ParticleState`

        Parameters
        ----------
        params : dict, optional
            Keyword arguments for :attr:`distribution_func`. These parameters will update the
            parameters specified in the class properties and can either be completely redefined
            or the subset of parameters that need changing.
        timestamp : datetime.datetime, optional
            Timestamp for the returned :class:`~.ParticleState`. Default is ``None``.

        Returns
        -------
        particle state : :class:`~.ParticleState`
            The particle state containing the samples of the distribution
            """

        if params is not None:
            params_update = params
            params = self.params.copy()
            params.update(**params_update)
        else:
            params = self.params.copy()

        samples = self.distribution_func(**params)
        # If samples is 1D, make it 2D
        if len(np.shape(samples)) == 1:
            samples = np.array([samples])
        # get the number of samples returned
        nsamples = (set(np.shape(samples)) - set(np.array([self.ndim_state]))).pop()

        # Ensure the correct shape of samples for the state_vector
        if np.shape(samples)[0] != self.ndim_state:
            samples = samples.T

        particles = ParticleState(state_vector=StateVectors(samples),
                                  weight=np.array([1 / nsamples] * nsamples),
                                  timestamp=timestamp)

        return particles
