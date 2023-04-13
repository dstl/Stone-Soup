import numpy as np

from .base import Sampler
from ..base import Property
from typing import Callable
from ..types.state import ParticleState
from ..types.array import StateVectors


class ParticleSampler(Sampler):

    distribution_func: Callable = Property(
        doc="Callable function that returns samples from the desired distribution.")

    params: dict = Property(
        doc="Dictionary containing the keyword arguments for `distribution_func`.")

    ndim_state: int = Property(
        doc="Number of dimensions in each sample.")

    def sample(self, distribution_func=None, params=None, ndim_state=None, timestamp=None):
        """Samples from the desired distribution and returns as a :class:`~.ParticleState`

        Parameters
        ----------
        distribution_func : :class:`~.Callable`, optional
            Callable function that returns samples from the desired distribution. Overrides the
            class property `distribution_func`.
        params : dict, optional
            Keyword arguments for `distribution_func`. These parameters will update the
            parameters specified in the class properties and can either be completely redefined
            or the subset of parameters that need changing. In the case where `distribution_func`
            is defined here, params needs to be completely specified as the parameters and values
            must be compatible with the new function.
        ndim_state : int, optional
            Number of dimensions in the state. Overrides the class property `ndim_state`.
            Only used when `distribution_func` is defined here.
        timestamp : datetime.datetime, optional
            Timestamp for the returned :class:`~.ParticleState`. Default is ``None``.

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
        # If samples is 1D, make it 2D
        if len(np.shape(samples)) == 1:
            samples = np.array([samples])
        # get the number of samples returned
        nsamples = (set(np.shape(samples)) - set(np.array([ndim_state]))).pop()

        # Ensure the correct shape of samples for the state_vector
        if np.shape(samples)[0] != ndim_state:
            samples = samples.T

        particles = ParticleState(state_vector=StateVectors(samples),
                                  weight=np.array([1 / nsamples] * nsamples),
                                  timestamp=timestamp)

        return particles
