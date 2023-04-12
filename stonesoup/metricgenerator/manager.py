from itertools import chain
from typing import Sequence, Iterable, Union

from .base import MetricManager, MetricGenerator
from ..base import Property
from ..dataassociator import Associator
from ..platform import Platform
from ..types.detection import Detection
from ..types.groundtruth import GroundTruthPath
from ..types.track import Track


class SimpleManager(MetricManager):
    """SimpleManager class for metric management

    Simple :class:`~.MetricManager` for the generation of metrics on multiple
    :class:`~.Track`, :class:`~.Detection` and :class:`~.GroundTruthPath`
    objects.
    """
    generators: Sequence[MetricGenerator] = Property(doc='List of generators to use', default=None)
    associator: Associator = Property(doc="Associator to combine tracks and truth", default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracks = set()
        self.groundtruth_paths = set()
        self.detections = set()
        self.association_set = None

    def add_data(self, groundtruth_paths: Iterable[Union[GroundTruthPath, Platform]] = None,
                 tracks: Iterable[Track] = None, detections: Iterable[Detection] = None,
                 overwrite=True):
        """Adds data to the metric generator

        Parameters
        ----------
        groundtruth_paths : list or set of :class:`~.GroundTruthPath`
            Ground truth paths to be added to the manager.
        tracks : list or set of :class:`~.Track`
            Tracks objects to be added to the manager.
        detections : list or set of :class:`~.Detection`
            Detections to be added to the manager.
        overwrite: bool
            declaring whether pre-existing data will be overwritten. Note that
            overwriting one field (e.g. tracks) does not affect the others
        """

        self._add(overwrite, groundtruth_paths=groundtruth_paths,
                  tracks=tracks, detections=detections)

    def _add(self, overwrite, **kwargs):
        for key, value in kwargs.items():
            if value is not None:
                if overwrite:
                    setattr(self, key, set(value))
                else:
                    getattr(self, key).update(value)

    def associate_tracks(self):
        """Associate tracks to truth using the associator

        The resultant :class:`~.AssociationSet` internally.
        """
        self.association_set = self.associator.associate_tracks(
            self.tracks, self.groundtruth_paths)

    def generate_metrics(self):
        """Generate metrics using the generators and data that has been added

        Returns
        ----------
        : set of :class:`~.Metric`
            Metrics generated
        """

        if self.associator is not None and self.association_set is None:
            self.associate_tracks()

        metrics = {}
        for generator in self.generators:
            metric_list = generator.compute_metric(self)
            # If not already a list, force it to be one below
            if not isinstance(metric_list, list):
                metric_list = [metric_list]
            for metric in metric_list:
                metrics[metric.title] = metric
        return metrics

    def list_timestamps(self):
        """List all the timestamps used in the tracks and truth, in order

        Returns
        ----------
        : list of :class:`datetime.datetime`
            unique timestamps present in the internal tracks and truths.
        """
        # Make a list of all the unique timestamps used
        timestamps = {state.timestamp
                      for sequence in chain(self.tracks, self.groundtruth_paths)
                      for state in sequence}

        return sorted(timestamps)
