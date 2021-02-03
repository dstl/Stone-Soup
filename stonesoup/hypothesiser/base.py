# -*- coding: utf-8 -*-
from ..base import Base


class Hypothesiser(Base):
    """Hypothesiser base class

    Given a track and set of detections, generate hypothesis of association.
    """

    def hypothesise(self, track, detections, timestamp, **kwargs):
        """Hypothesise track and detection association

        Parameters
        ----------
        track : Track
            Track which hypotheses will be generated for.
        detections :
            Detections used to generate hypotheses.
        timestamp: :class:`datetime.datetime`
            A timestamp used when evaluating the state and measurement
            predictions. Note that if a given detection has a non empty
            timestamp, then prediction will be performed according to
            the timestamp of the detection.


        Returns
        -------
        : sequence of :class:`~.Hypothesis`
            Ordered sequence of "best" to "worse" hypothesis.
        """
        raise NotImplementedError
