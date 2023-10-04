from abc import abstractmethod

import numpy as np

from .base import Sampler
from ..base import Property
from ..models.measurement.linear import LinearModel
from ..functions import jacobian, gm_sample
from ..types.state import ParticleState, StateVectors


class DetectionSampler(Sampler):
    """Detection sampler base class.

    Samples from a continuous distribution based on provided detections.
    """
    @abstractmethod
    def sample(self, detections):
        """Sample from continuous distribution based on detections

        Parameters
        ----------
        detections : set of :class:`~.Detection`
            Detections used to describe the distribution

        Returns
        -------
        :class:`~.State`
        """

        raise NotImplementedError


class GaussianDetectionParticleSampler(DetectionSampler):
    """Particle sampler using Gaussian detections to initialise the distribution.

    Particle sampler that is preloaded with the :func:`~.functions.gm_sample` method for sampling
    from Gaussian mixture distributions. This class can handle one or more linear and non-linear
    Gaussian detections and will either return samples from a single or mixture of Gaussians
    depending on which is provided."""

    nsamples: int = Property(
        default=1,
        doc="Number of samples to return")

    def sample(self, detections):
        """Samples from a Gaussian mixture around detections

        Parameters
        ----------
        detections : :class:`~.Detection` or set of :class:`~.Detection`
            Detections forming the distribution to sample from. Equal component weighting is
            assumed.
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

        samples = gm_sample(means=dist_mean,
                            covars=dist_covar,
                            weights=weights,
                            size=self.nsamples)

        particles = ParticleState(state_vector=StateVectors(samples),
                                  weight=np.array([1 / self.nsamples] * self.nsamples),
                                  timestamp=timestamp)
        return particles

    @staticmethod
    def get_weight(num_detections):

        weights = np.array([1/num_detections]*num_detections)

        return weights


class SwitchingDetectionSampler(DetectionSampler):
    """Redundant detection sampler class.

    Redundant detection sampler accepts two :class:`~.Sampler` objects, one
    :class:`~.DetectionSampler` and one to fall back on when detections are not available. The
    samples returned depend on which samplers have been specified. Both samplers must have a
    :meth:`sample` method.
    """

    detection_sampler: DetectionSampler = Property(
        doc="Sampler for generating samples from detections")

    backup_sampler: Sampler = Property(
        doc="Sampler for generating samples in the absence of detections")

    def sample(self, detections, timestamp=None):
        """Produces samples based on the detections provided.

        Parameters
        ----------
        detections : :class:`~.Detection` or set of :class:`~.Detection`
            Detections used for generating samples.
        timestamp : datetime.datetime, optional
            timestamp for when the sample is made.

        Returns
        -------
        : :class:`~.State`
            The state object returned by either of the specified samplers.
        """

        if detections:
            sample = self.detection_sampler.sample(detections)

        else:
            sample = self.backup_sampler.sample(timestamp=timestamp)

        return sample
