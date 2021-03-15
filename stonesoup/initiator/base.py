# -*- coding: utf-8 -*-
from abc import abstractmethod

from ..base import Base


class Initiator(Base):
    """Initiator base class

    Creates zero or more tracks based on provided detections.
    """

    @abstractmethod
    def initiate(self, detections, timestamp, **kwargs):
        """Generate tracks from detections.

        Parameters
        ----------
        detections : set of :class:`~.Detection`
            Detections used to generate set of tracks
        timestamp: datetime.datetime
            Current timestamp

        Returns
        -------
        : set of :class:`~.Track`
            Tracks generated from detections
        """
        raise NotImplementedError


class GaussianInitiator(Initiator):
    """Gaussian Initiator base class

    Base class for initiator's which initialises tracks with a
    :class:`~.GaussianState`
    """


class ParticleInitiator(Initiator):
    """Particle Initiator base class

    Base class for initiator's which initialises tracks with a
    :class:`~.ParticleState`
    """
