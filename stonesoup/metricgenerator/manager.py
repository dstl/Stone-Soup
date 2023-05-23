from itertools import chain
from typing import Sequence, Dict

from .base import MetricManager, MetricGenerator
from ..base import Property
from ..dataassociator import Associator
from .tracktotruthmetrics import SIAPMetrics


class MultiManager(MetricManager):
    """MultiManager class for metric management

    :class:`~.MetricManager` for the generation of metrics on multiple sets of
    :class:`~.Track`, :class:`~.Detection` and :class:`~.GroundTruthPath`
    objects passed in as dictionaries.
    """
    generators: Sequence[MetricGenerator] = Property(doc='List of generators to use', default=None)
    associator: Associator = Property(doc="Associator to combine tracks and truth", default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.states_sets = dict()
        self.association_set = None

    def add_data(self, metric_data: Dict = None, overwrite=True):
        """Adds data to the metric generator

        Parameters
        ----------
        metric_data : dict of lists or dict of sets of :class:`~.GroundTruthPath`, :class:`~.Track`, and/or
        :class:`~.Detection`
            Ground truth paths, Tracks, and/or detections to be added to the manager.
        overwrite: bool
            declaring whether pre-existing data will be overwritten. Note that
            overwriting one field (e.g. tracks) does not affect the others
        """
        self._add(overwrite, metric_data=metric_data)

    def _add(self, overwrite, metric_data):
        if overwrite:
            self.states_sets = metric_data
        else:
            for key, value in metric_data.items():
                if key not in self.states_sets.keys():
                    self.states_sets[key] = set(value)
                else:
                    self.states_sets[key].update(value)

    def associate_tracks(self, generator):
        """Associate tracks to truth using the associator

        The resultant :class:`~.AssociationSet` internally.
        """
        self.association_set = self.associator.associate_tracks(
            self.states_sets[generator.tracks_key], self.states_sets[generator.truths_key])

    def generate_metrics(self):
        """Generate metrics using the generators and data that has been added

        Returns
        ----------
        : set of :class:`~.Metric`
            Metrics generated
        """

        metrics = {}
        for generator in self.generators:
            if isinstance(generator, SIAPMetrics):
                if self.associator is not None and self.association_set is None:
                    self.associate_tracks(generator)
            metric_list = generator.compute_metric(self)
            # If not already a list, force it to be one below
            if not isinstance(metric_list, list):
                metric_list = [metric_list]
            for metric in metric_list:
                if generator.generator_name not in metrics.keys():
                    metrics[generator.generator_name] = {metric.title: metric}
                else:
                    metrics[generator.generator_name][metric.title] = metric

        return metrics

    def list_timestamps(self, generator):
        """List all the timestamps used in the tracks and truth, in order

        Returns
        ----------
        : list of :class:`datetime.datetime`
            unique timestamps present in the internal tracks and truths.
        """
        # Make a list of all the unique timestamps used
        timestamps = {state.timestamp
                      for sequence in chain(self.states_sets[generator.tracks_key],
                                            self.states_sets[generator.truths_key])
                      for state in sequence}

        return sorted(timestamps)
