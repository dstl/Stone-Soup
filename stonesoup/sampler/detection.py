from .base import Sampler
from ..base import Property


class DetectionSampler(Sampler):

    detection_sampler: Sampler = Property(
        doc="Sampler for generating samples from detections")

    backup_sampler: Sampler = Property(
        doc="Sampler for generating samples in the absence of detections")

    def sample(self, detections, timestamp=None):
        """Produces samples based on the detections provided.

        Parameters
        ----------
        detections : :class:`~.Detection` or set of :class:`~.Detection`
            detections used for generating samples. Assumed to follow a gaussian distribution.
        timestamp : datetime.datetime, optional
            timestamp for when the sample is made.

        Returns
        -------
        : :class:`~.State`
            The state object returned by either of the defined samplers.
        """

        if detections:
            sample = self.detection_sampler.sample(detections)

        else:
            sample = self.backup_sampler.sample(timestamp=timestamp)

        return sample
