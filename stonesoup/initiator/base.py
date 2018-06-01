# -*- coding: utf-8 -*-
from abc import abstractmethod

from ..base import Base


class Initiator(Base):
    """Initiator base class

    Creates zero or more tracks based on provided detections.
    """

    @abstractmethod
    def initiate(self, detections, **kwargs):
        """Generate tracks from detections.

        Parameters
        ----------
        detections : set of :class:`~.Detection`
            Detections used to generate set of tracks

        Returns
        -------
        : set of :class:`~.Track`
            Tracks generated from detections
        """
        raise NotImplemented
