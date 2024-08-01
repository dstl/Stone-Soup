import datetime
from typing import Set, Sequence

from ..base import Base
from ..types.detection import Detection
from ..types.hypothesis import Hypothesis
from ..types.track import Track


class Hypothesiser(Base):
    """Hypothesiser base class

    Given a track and set of detections, generate hypothesis of association.
    """

    def hypothesise(self, track: Track, detections: Set[Detection], timestamp: datetime.datetime,
                    **kwargs) -> Sequence[Hypothesis]:
        """Hypothesise track and detection association

        Parameters
        ----------
        track : Track
            Track which hypotheses will be generated for.
        detections : set of :class:`~.Detection`
            Detections used to generate hypotheses.
        timestamp : datetime.datetime
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
