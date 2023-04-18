import datetime
from abc import abstractmethod
from typing import Set, Mapping

from ..base import Base, Property
from ..hypothesiser import Hypothesiser
from ..types.detection import Detection
from ..types.hypothesis import Hypothesis
from ..types.track import Track
from ..types.association import AssociationSet


class DataAssociator(Base):
    """Data Associator base class

    A data associator is used to associate tracks and detections, and may also
    include an association of a missed detection. The associations generate are
    in the form a mapping each track to a hypothesis, based on "best" choice
    from hypotheses generate from a :class:`~.Hypothesiser`.
    """

    hypothesiser: Hypothesiser = Property(
        doc="Generate a set of hypotheses for each track-detection pair")

    def generate_hypotheses(self, tracks, detections, timestamp, **kwargs):
        return {track: self.hypothesiser.hypothesise(
                    track, detections, timestamp, **kwargs)
                for track in tracks}

    @abstractmethod
    def associate(self, tracks: Set[Track], detections: Set[Detection],
                  timestamp: datetime.datetime, **kwargs) -> Mapping[Track, Hypothesis]:
        """Associate tracks and detections

        Parameters
        ----------
        tracks : set of :class:`~.Track`
            Tracks which detections will be associated to.
        detections : set of :class:`~.Detection`
            Detections to be associated to tracks.
        timestamp : datetime.datetime
            Timestamp to be used for missed detections and to predict to.

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
    """Associates 'n' sets of :class:`~.Track` objects together"""

    @abstractmethod
    def associate_tracks(self, *tracks_sets: Set[Track]) \
            -> AssociationSet:
        """Associate n sets of tracks together.

        Parameters
        ----------
        tracks_sets : n sets of :class:`~.Track` objects
            Tracks to associate to other track sets

        Returns
        -------
        AssociationSet
            Contains a set of :class:`~.Association` objects

        """
