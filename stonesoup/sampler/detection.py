from .base import Sampler
from ..base import Property


class DetectionSampler(Sampler):
    """Detection sampler class.

    Detection sampler accepts two :class:`~.Sampler` objects, one intended for use when
    detections are available and one to fall back on when detections are not available. The
    samples returned depend on which samplers have been specified. Both samplers must have a
    :meth:`sample` method.
    """

    detection_sampler: Sampler = Property(
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
