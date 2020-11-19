# -*- coding: utf-8 -*-
from itertools import chain
from typing import Sequence

from .base import MetricManager, MetricGenerator
from ..base import Property
from ..dataassociator import Associator
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

    def add_data(self, input_objects, overwrite=True):
        """Adds data to the metric generator

        Parameters
        ----------
        input_objects : list or set of objects
            Objects to be added to the manager. The class of the object is used
            to determine whether it is track, truth or detection
        overwrite: bool
            declaring whether pre-existing data will be overwritten. Note that
            overwriting one field (e.g. tracks) does not affect the others
        """
        for in_obj in input_objects:
            if not isinstance(in_obj, (list, set)):
                raise TypeError('Inputs are expected as lists or sets only')
            else:
                if all(isinstance(x, Track) for x in in_obj):
                    if overwrite:
                        self.tracks = set(in_obj)
                    else:
                        self.tracks = self.tracks.union(set(in_obj))
                elif all(isinstance(x, GroundTruthPath)
                         for x in in_obj):
                    if overwrite:
                        self.groundtruth_paths = set(in_obj)
                    else:
                        self.groundtruth_paths = self.groundtruth_paths.union(
                            set(in_obj))
                elif all(isinstance(x, Detection) for x in in_obj):
                    if overwrite:
                        self.detections = set(in_obj)
                    else:
                        self.detections = self.detections.union(set(in_obj))
                else:
                    raise TypeError(
                        'Object of type {!r} not expected'.format(type()))
                    # This error doesn't work if the first element of the list
                    # is a sensible one but the later ones aren't.

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
        return sorted(timestamps)
