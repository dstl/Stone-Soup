import numpy as np

from .base import Sampler
from ..base import Property
from typing import Callable
from ..types.state import ParticleState
from ..types.array import StateVectors


class ParticleSampler(Sampler):

    distribution_func: Callable = Property(
        doc="Callable function that samples from the desired distribtuion.")

    params: dict = Property(
        doc="Dictionary containing the key word arguments to the `distribtuion_func`.")

    ndim_state: int = Property(
        doc="Number of dimensions in each sample.")

    def sample(self, distribution_func=None, params=None, ndim_state=None, timestamp=None):
        """Sample particles from the distribution

        Parameters
        ----------
        distribution_func : :class:`~.Callable`, optional
            Function defining the distribution to be sampled from. Overrides the class
            property `distribution_func`.
        params : dict, optional
            Pairs of `parameters and values to be input to `distribution_func`. These parameters
            will update the parameters specified in the class properties and can either be
            completely redefined or the subset of parameters that need changing. In the case where
            `distribution_func` is defined here, params needs to be completely specified as the
            parameters and values must be compatible with the new function.
        ndim_state : int, optional
            Number of dimensions in the state. Overrides the class property `ndim_state`.
            Only used when `distribution_func` is defined here.
        timestamp : datetime.datetime, optional
            Timestamp for the returned :class:`~.ParticleState`

        Returns
        -------
        particle state : :class:`~.ParticleState`
            The particle state containing the samples of the distribution"""

        if distribution_func is None and params is not None:
            distribution_func = self.distribution_func
            params_update = params
            params = self.params.copy()
            params.update(**params_update)
            ndim_state = self.ndim_state
        elif distribution_func is None:
            distribution_func = self.distribution_func
            params = self.params.copy()
            ndim_state = self.ndim_state
        elif params is None:
            raise ValueError('New distribution_func provided without params')
        elif ndim_state is None:
            raise ValueError('New distribution_func provided without ndim_state')

        samples = distribution_func(**params)
        if len(np.shape(samples)) == 1:
            samples = np.array([samples])
        nsamples = (set(np.shape(samples)) - set(np.array([ndim_state]))).pop()

        if np.shape(samples)[0] != ndim_state:
            samples = samples.T

        particles = ParticleState(state_vector=StateVectors(samples),
                                  weight=np.array([1 / nsamples] * nsamples),
                                  timestamp=timestamp)

        return particles
