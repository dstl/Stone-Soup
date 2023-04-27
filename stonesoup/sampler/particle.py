import numpy as np

from .base import Sampler
from ..base import Property
from typing import Callable
from ..types.state import ParticleState
from ..types.array import StateVectors
from ..models.measurement.linear import LinearModel
from ..functions import jacobian, gm_sample


class ParticleSampler(Sampler):

    distribution_func: Callable = Property(
        doc="Callable function that returns samples from the desired distribution.")

    params: dict = Property(
        doc="Dictionary containing the keyword arguments for :attr:`distribution_func`.")

    ndim_state: int = Property(
        doc="Number of dimensions in each sample.")

    def sample(self, distribution_func=None, params=None, ndim_state=None, timestamp=None):
        """Samples from the desired distribution and returns as a :class:`~.ParticleState`

        Parameters
        ----------
        distribution_func : :class:`~.Callable`, optional
            Callable function that returns samples from the desired distribution. Overrides the
            class property :attr:`distribution_func`.
        params : dict, optional
            Keyword arguments for :attr:`distribution_func`. These parameters will update the
            parameters specified in the class properties and can either be completely redefined
            or the subset of parameters that need changing. In the case where
            :attr:`distribution_func` is defined here, params needs to be completely specified as
            the parameters and values must be compatible with the new function.
        ndim_state : int, optional
            Number of dimensions in the state. Overrides the class property :attr:`ndim_state`.
            Only used when :attr:`distribution_func` is defined here.
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


class GaussianDetectionParticleSampler(ParticleSampler):

    distribution_func = None
    params = None
    ndim_state = None

    nsamples: int = Property(
        default=1,
        doc="Number of samples to return")

    def sample(self, detections):
        """Samples from a gaussian mixture around detections

        Parameters
        ----------
        detections : :class:`~.Detection` or set of :class:`~.Detection`
            Detections forming the distribution to sample from.
        Returns
        -------
        : :class:`~.ParticleState`
            :class:`~.ParticleState` of samples."""

        dist_mean = []
        dist_covar = []
        num_det = len(detections)
        timestamp = next(iter(detections)).timestamp
        for detection in detections:
            ndim_state = detection.measurement_model.ndim_state
            ndim_meas = detection.measurement_model.ndim
            if isinstance(detection.measurement_model, LinearModel):
                if ndim_state > ndim_meas:
                    mapping = detection.measurement_model.mapping
                    mapping_matrix = np.zeros((ndim_state, ndim_meas))
                    mapping_index = np.linspace(0, len(mapping)-1, ndim_meas, dtype=int)
                    mapping_matrix[mapping, mapping_index] \
                        = 1
                    dist_mean.append(mapping_matrix @ detection.state_vector)
                    dist_covar.append(mapping_matrix @
                                      detection.measurement_model.noise_covar @
                                      mapping_matrix.T)
                else:
                    dist_mean.append(detection.state_vector)
                    dist_covar.append(detection.measurement_model.noise_covar)
            else:
                tmp_mean = detection.measurement_model.inverse_function(detection)
                jac = jacobian(detection.measurement_model.inverse_function, detection)
                tmp_covar = jac @ detection.measurement_model.noise_covar @ jac.T
                dist_mean.append(tmp_mean)
                dist_covar.append(tmp_covar)

        weights = self.get_weight(num_det)

        params = {'means': dist_mean,
                  'covars': dist_covar,
                  'weights': weights,
                  'size': self.nsamples}
        samples = super().sample(distribution_func=gm_sample,
                                 params=params,
                                 ndim_state=ndim_state,
                                 timestamp=timestamp)

        return samples

    @staticmethod
    def get_weight(num_detections):

        weights = np.array([1/num_detections]*num_detections)

        return weights
