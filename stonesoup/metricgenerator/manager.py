# -*- coding: utf-8 -*-
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
        self._tracks = set()
        self._groundtruth_paths = set()
        self._detections = set()
        self.association_set = None
        if self.generators is None:
            self.generators = set()

    @property
    def tracks(self):
        return frozenset(self._tracks)

    @property
    def groundtruth_paths(self):
        return frozenset(self._groundtruth_paths)

    @property
    def detections(self):
        return frozenset(self._detections)

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

        # Clear all generator caches
        for generator in self.generators:
            try:
                generator.clear_caches()
            except AttributeError:
                pass

        self._add(overwrite, _groundtruth_paths=groundtruth_paths,
                  _tracks=tracks, _detections=detections)

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

        metrics = set()
        for generator in self.generators:
            metric = generator.compute_metric(self)
            # Metrics can be lists or not, there's probably a neater way to do
            # this
            if isinstance(metric, list):
                metrics.update(metric)
            else:
                metrics.add(metric)

        return set(metrics)

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
        return tuple(sorted(timestamps))
