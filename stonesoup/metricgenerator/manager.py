from itertools import chain
from typing import Sequence, Dict, Iterable, Union

from .base import MetricManager, MetricGenerator
from ..base import Property
from ..dataassociator import Associator
from .basicmetrics import BasicMetrics

from ..types.groundtruth import GroundTruthPath
from ..types.track import Track
from ..types.detection import Detection
from ..platform import Platform


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
        self.metrics = None

    def add_data(self, metric_data: Dict = None, overwrite=True):
        """Adds data to the metric generator

        Parameters
        ----------
        metric_data : dict of lists or dict of sets of :class:`~.GroundTruthPath`, \
        :class:`~.Track`, and/or :class:`~.Detection`
            Ground truth paths, Tracks, and/or detections to be added to the manager.
        overwrite: bool
            Declaring whether pre-existing data will be overwritten. Note that
            overwriting one key-value pair (e.g. 'tracks') does not affect the others.

        """
        self._add(overwrite, metric_data=metric_data)

    def _add(self, overwrite, metric_data):
        if overwrite:
            for key, value in metric_data.items():
                self.states_sets[key] = set(value)
        else:
            for key, value in metric_data.items():
                if key not in self.states_sets.keys():
                    self.states_sets[key] = set(value)
                else:
                    self.states_sets[key].update(value)

    def associate_tracks(self, generator):
        """Associate tracks to truth using the associator to produce an
         :class:`~.AssociationSet`

        Parameters
        ----------
        generator : :class:`~.MetricGenerator`
            :class:`~.MetricGenerator` containing `tracks_key` and `truths_key` to extract
            tracks and truths from :class:`~.MetricManager` for association.
        """
        self.association_set = self.associator.associate_tracks(
            self.states_sets[generator.tracks_key], self.states_sets[generator.truths_key])

    def _get_metrics(self):
        return self.metrics

    def generate_metrics(self):
        """Generate metrics using the generators and data that has been added

        Returns
        -------
        : nested dict of :class:`~.Metric`
            Metrics generated
        """

        metrics = {}

        generators = self.generators if isinstance(self.generators, list) else [self.generators]

        for generator in generators:
            if self.associator is not None and \
                    hasattr(generator, 'tracks_key') and hasattr(generator, 'truths_key'):
                self.associate_tracks(generator)
            metric_list = generator.compute_metric(self)
            if not isinstance(metric_list, list):  # If not already a list, force it to be one
                metric_list = [metric_list]
            for metric in metric_list:
                if generator.generator_name not in metrics.keys():
                    metrics[generator.generator_name] = {metric.title: metric}
                else:
                    metrics[generator.generator_name][metric.title] = metric

        self.metrics = metrics

        return self._get_metrics()

    def list_timestamps(self, generator=None):
        """List all the unique timestamps used in the tracks and truth associated
        with a given generator, in order

        Parameters
        ----------
        generator : :class:`~.MetricGenerator`
            :class:`~.MetricGenerator` containing `tracks_key` and `truths_key` to extract
            tracks and truths from :class:`~.MetricManager` to extract timestamps from.
            Default None to take tracks and truths values from first :class:`~.MetricGenerator`
            in `self.generators`.

        Returns
        -------
        : list of :class:`datetime.datetime`
            unique timestamps present in the internal tracks and truths.
        """
        if generator is None:
            generator = self.generators[0]
        timestamps = {state.timestamp
                      for sequence in chain(self.states_sets[generator.tracks_key],
                                            self.states_sets[generator.truths_key])
                      for state in sequence}

        return sorted(timestamps)

    def display_basic_metrics(self):
        """
        Print basic metrics generated for each :class:`BasicMetrics` generator.
        """
        for generator_name, generator in self.metrics.items():
            if "Number of targets" in generator.keys():
                print(f'\nGenerator: {generator_name}')
                for metric_key, metric in generator.items():
                    if isinstance(metric.generator, BasicMetrics):
                        print(f"{metric.title}: {metric.value}")

    def get_siap_averages(self, generator_name):
        """
        Get SIAP averages metrics from SIAP metric generator specified by generator_name

        Parameters
        ----------
        generator_name : str
            Name of SIAP :class:`~.MetricGenerator`

        Returns
        -------
        : dict of :class`SIAPMetrics` averages
        """
        siap_metrics = self.metrics[generator_name]
        siap_averages = {siap_metrics.get(metric) for metric in siap_metrics
                         if metric.startswith("SIAP") and not metric.endswith(" at times")}

        return siap_averages


class SimpleManager(MultiManager):
    """SimpleManager class for metric management

    Simple :class:`~.MetricManager` for the generation of metrics on multiple
    :class:`~.Track`, :class:`~.Detection` and :class:`~.GroundTruthPath`
    objects.
    """
    generators: Sequence[MetricGenerator] = Property(doc='List of generators to use', default=None)
    associator: Associator = Property(doc="Associator to combine tracks and truth", default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.states_sets = dict()
        self.association_set = None
        self.metrics = None

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
        if overwrite:
            for key, value in kwargs.items():
                if value is not None:
                    self.states_sets[key] = set(value)
        else:
            for key, value in kwargs.items():
                if value is not None:
                    if key not in self.states_sets.keys():
                        self.states_sets[key] = set(value)
                    else:
                        self.states_sets[key].update(value)

    def _get_metrics(self):
        metrics = {}
        for key, value in self.metrics.items():
            metrics.update(value)

        return metrics
