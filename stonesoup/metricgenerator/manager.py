from .base import MetricManager, PlotGenerator
from ..base import Property
from ..types import Detection, GroundTruthPath, Track


class SimpleManager(MetricManager):
    """
    Parses the output of the tracking methods to and calls a series of MetricGenerators to produce Metrics
    """

    generators = Property(list, doc='List of generators to use', default=None)

    def __init__(self, generators, *args, **kwargs):
        super().__init__(generators, *args, **kwargs)
        self.tracks = []
        self.groundtruth_paths = []
        self.detections = []

    def add_data(self, input_objects, overwrite = True):
        """
        Adds data to the metric generator
        :param input_objects: list, set or tuple of objects to be added. The objects are lists or sets of Tracks,
         GroundTruthPaths or Detections. e.g. manager.add_data((tracks,detections))
        :param overwrite: If true then the existing data is overwritten with the new, if False then any elemts of the new
        list not existing in the old are added
        :return:
        """
        for in_obj in input_objects:
            if type(in_obj) != list and type(in_obj) != set:
                raise TypeError('Inputs are expected as lists or sets only')
            else:
                if all(isinstance(x, Track) or issubclass(x, Track) for x in in_obj):

                    if overwrite:
                        self.tracks = set(in_obj)
                    else:
                        self.tracks = self.tracks.union(set(in_obj))


                elif all(isinstance(x, GroundTruthPath) or issubclass(x, GroundTruthPath) for x in in_obj):

                    if not self.groundtruth_paths:
                        self.groundtruth_paths = in_obj
                    else:
                        raise ValueError('Only one list or set of GroundTruthPath objects expected')

                elif all(isinstance(x, Detection) or issubclass(x, Detection) for x in in_obj):

                    if not self.detections:
                        self.detections = in_obj
                    else:
                        raise ValueError('Only one list or set of Detection objects expected')

                else:
                    raise TypeError('Object of type "' + type(in_obj[0] + '" not expected'))
                    # This error doesn't work if the first element of the list is a sensible one but the later ones aren't.

    def generate_metrics(self):
        """

        :param tracks: set of Track objects created by the tracker
        :param groundtruth_paths:  set of GroundTruthPath objects
        :param detections: set of list of Detection objects
        :return: set of Metric objects (or objects that inherit from metric)
        """
        metrics = []
        n_plotting_metrics = len([i for i in self.generators if isinstance(i, PlotGenerator)])
        i_plot = 1
        for generator in self.generators:

            metric = generator.compute_metric(self)
            # Metrics can be lists or not, there's probably a neater way to do this

            if type(metric) == list:
                metrics += metric
            else:
                metrics.append(metric)

        return metrics
