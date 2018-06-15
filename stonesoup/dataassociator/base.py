# -*- coding: utf-8 -*-
import itertools
from abc import abstractmethod

from ..base import Base, Property
from ..types import JointHypothesis
from ..hypothesiser import Hypothesiser


class DataAssociator(Base):
    """Data Associator base class

    A data associator is used to associate tracks and detections, and may also
    include an association of a missed detection. The associations generate are
    in the form a mapping each track to a hypothesis, based on "best" choice
    from hypotheses generate from a :class:`~.Hypothesiser`.
    """

    hypothesiser = Property(
        Hypothesiser,
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
        timestamp : :class:`datetime.datetime`
            Timestamp to be used for missed detections.

        Returns
        -------
        : mapping of :class:`~.Track` : :class:`~.Hypothesis`}
            Mapping of track to Hypothesis
        """
        raise NotImplemented

    @staticmethod
    def isvalid(joint_hypothesis):
        """Determine whether a joint_hypothesis is valid.

        Check the set of hypotheses that define a joint hypothesis to ensure a
        single detection is not associated to more than one track.

        Parameters
        ----------
        joint_hypothesis : :class:`JointHypothesis`
            A set of hypotheses linking each prediction to a single detection

        Returns
        -------
        bool
            Whether joint_hypothesis is a valid set of hypotheses
        """

        number_hypotheses = len(joint_hypothesis)
        unique_hypotheses = len(
            {hyp.detection for hyp in joint_hypothesis} - {None})
        number_null_hypotheses = sum(
            hyp.detection is None for hyp in joint_hypothesis)

        # joint_hypothesis is invalid if one detection is assigned to more than
        # one prediction. Multiple missed detections are valid.
        if unique_hypotheses + number_null_hypotheses == number_hypotheses:
            return True
        else:
            return False

    @classmethod
    def enumerate_joint_hypotheses(cls, hypotheses):
        """Enumerate the possible joint hypotheses.

        Create a list of all possible joint hypotheses from the individual
        hypotheses and determine whether each is valid.

        Parameters
        ----------
        hypotheses : list of :class:`Hypothesis`
            A list of all hypotheses linking predictions to detections,
            including missed detections

        Returns
        -------
        joint_hypotheses : list of :class:`JointHypothesis`
            A list of all valid joint hypotheses with a score on each
        """

        # Create a list of dictionaries of valid track-hypothesis pairs
        joint_hypotheses = [
            JointHypothesis({
                track: hypothesis
                for track, hypothesis in zip(hypotheses, joint_hypothesis)})
            for joint_hypothesis in itertools.product(*hypotheses.values())
            if cls.isvalid(joint_hypothesis)]

        return joint_hypotheses
