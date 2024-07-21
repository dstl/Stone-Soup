from ..base import Property
from ..types.association import AssociationSet
from ..types.metric import TimeRangeMetric
from ..types.time import TimeRange
from .base import MetricGenerator
from .manager import MultiManager


class CLEARMOTMetrics(MetricGenerator):
    """TODO"""
    tracks_key: str = Property(doc='Key to access set of tracks added to MetricManager',
                               default='tracks')
    truths_key: str = Property(doc="Key to access set of ground truths added to MetricManager. "
                                   "Or key to access a second set of tracks for track-to-track "
                                   "metric generation",
                               default='groundtruth_paths')

    def compute_metric(self, manager: MultiManager, **kwargs):

        timestamps = manager.list_timestamps(generator=self)
        tracks = self._get_data(manager, self.tracks_key)
        ground_truths = self._get_data(manager, self.truths_key)

        # TODO: CODE HERE

        time_range = TimeRange(min(timestamps), max(timestamps))

        motp = TimeRangeMetric(title="MOTP",
                               value=0.0,
                               time_range=time_range,
                               generator=self)
        mota = TimeRangeMetric(title="MOTA",
                               value=0.0,
                               time_range=time_range,
                               generator=self)
        return [motp, mota]
    
    def compute_mota(self, manager: MultiManager) -> float:
        # TODO: WIP
        associations: AssociationSet = self.manager.association_set
        associations.associations_including_objects
        return 0.0

    @staticmethod
    def num_associated_truths_at_time(manager: MultiManager, ground_truths, timestamp):
        """:math:`JT(t)`. Calculate the number of associated true objects held by `manager` at
        `timestamp`.

        Parameters
        ----------
        manager: MetricManager
            Containing the data to be used
        ground_truths: set or list of :class:`~.GroundTruthPath` or :class:`~.Track` objects
            Containing the groundtruth or track data to be used
        timestamp: datetime.datetime
            Timestamp at which to compute the value

        Returns
        -------
        float
            Number of associated true objects held by `manager` at `timestamp`
        """
        associations = manager.association_set.associations_at_timestamp(timestamp)
        association_objects = {thing for assoc in associations for thing in assoc.objects}

        return sum(1 for truth in ground_truths if truth in association_objects) 