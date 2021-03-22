# -*- coding: utf-8 -*-
from abc import abstractmethod

from ..base import Base, Property
from ..hypothesiser import Hypothesiser


class DataAssociator(Base):
    """Data Associator base class

    A data associator is used to associate tracks and detections, and may also
    include an association of a missed detection. The associations generate are
    in the form a mapping each track to a hypothesis, based on "best" choice
    from hypotheses generate from a :class:`~.Hypothesiser`.
    """

    hypothesiser: Hypothesiser = Property(
        doc="Generate a set of hypotheses for each track-detection pair")

    @abstractmethod
    def associate(self, tracks, detections, timestamp=None, **kwargs):
        """Associate tracks and detections

        Parameters
        ----------
        tracks : set of :class:`~.Track`
            Tracks which detections will be associated to.
        detections : set of :class:`~.Detection`
            Detections to be associated to tracks.
        timestamp : datetime.datetime
            Timestamp to be used for missed detections.

        Returns
        -------
        : mapping of :class:`~.Track` : :class:`~.Hypothesis`
            Mapping of track to Hypothesis
        """
        raise NotImplementedError


class Associator(Base):
    """Associator base class

    An associator is used to associate objects for the generation of
    metrics. It returns a :class:`~.AssociationSet` containing
    a set of :class:`~.Association`
    objects.
    """


class TrackToTrackAssociator(Associator):
    """Associates two sets of :class:`~.Track` objects together"""
