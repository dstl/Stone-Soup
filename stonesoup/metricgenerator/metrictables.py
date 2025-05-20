from collections.abc import Collection
from operator import attrgetter

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from ..base import Property, Base
from .base import MetricGenerator, MetricTableGenerator


class RedGreenTableGenerator(MetricTableGenerator):
    """Red Green Metric Table Generator class

    Takes in a set of metrics and uses known ranges and target
    values to color code an output table with respect to how
    well the tracker performed in relation to each metric, where
    red is worse and green is better"""

    # needed because of MetricGenerator parent class
    generator_name: str = ""

    metrics: Collection[MetricGenerator] = Property(doc="Set of metrics to put in the table")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ranges = dict
        self.targets = dict
        self.descriptions = dict

    def compute_metric(self, **kwargs):
        """Generate table method

        Returns a matplotlib Table of metrics with their descriptions, target
        values and a coloured value cell to represent how well the tracker has
        performed in relation to each specific metric (red=bad, green=good)"""

        white = (1, 1, 1)
        cellText = [["Metric", "Description", "Target", "Value"]]
        cellColors = [[white, white, white, white]]

        for metric in sorted(self.metrics, key=attrgetter('title')):
            #  Add metric details to table row
            metric_name = metric.title
            description = self.descriptions[metric_name]
            target = self.targets[metric_name]
            value = metric.value
            cellText.append([metric_name, description, target, "{:.2f}".format(value)])

            # Generate color for value cell based on closeness to target value
            # Closeness to infinity cannot be represented as a color
            if target is not None and not target == np.inf:
                red_value = 1
                green_value = 1
                # A value of 1 for both red & green produces yellow

                metric_range = \
                    self.ranges[metric_name][1] - self.ranges[metric_name][0]
                closeness = abs(value - self.targets[metric_name]) * (1 / metric_range)

                if closeness > 1:  # Infinite range metric exceeded bound
                    closeness = 1
                if closeness > 0.5:  # Further away from target
                    green_value = closeness
                elif closeness < 0.5:  # Closer to target
                    red_value = closeness

                cellColors.append(
                    [white, white, white, (red_value, green_value, 0, 0.5)])
            else:
                cellColors.append([white, white, white, white])

        # "Plot" table
        scale = (1, 3)
        fig = plt.figure(figsize=(len(cellText)*scale[0] + 1, len(cellText[0])*scale[1]/2))
        ax = fig.add_subplot(1, 1, 1)
        ax.axis('off')
        table = matplotlib.table.table(ax, cellText, cellColors, loc='center')
        table.auto_set_column_width([0, 1, 2, 3])
        table.scale(*scale)

        return table


class SIAPTableGenerator(RedGreenTableGenerator):
    """Red Green Table Generator specifically for SIAP metrics.

    Contains methods to specify the ranges, descriptions and target values specifically for SIAP
    metrics."""

    # needed because of MetricGenerator parent class
    generator_name: str = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_default_ranges()
        self.set_default_targets()
        self.set_default_descriptions()

    def set_default_ranges(self):
        self.ranges = {
            "SIAP Completeness": (0, 1),
            "SIAP Ambiguity": (0, 10),  # True maximum is infinity
            # everything >10 will be treated as "worst"
            "SIAP Spuriousness": (0, 1),
            "SIAP Position Accuracy": (0, np.inf),
            "SIAP Velocity Accuracy": (0, np.inf),
            "SIAP Rate of Track Number Change": (0, np.inf),
            "SIAP Longest Track Segment": (0, 1),
            "SIAP ID Completeness": (0, 1),
            "SIAP ID Correctness": (0, 1),
            "SIAP ID Ambiguity": (0, 1)
        }

    def set_default_targets(self):
        self.targets = {
            "SIAP Completeness": 1,
            "SIAP Ambiguity": 1,
            "SIAP Spuriousness": 0,
            "SIAP Position Accuracy": 0,
            "SIAP Velocity Accuracy": 0,
            "SIAP Rate of Track Number Change": 0,
            "SIAP Longest Track Segment": 1,
            "SIAP ID Completeness": 1,
            "SIAP ID Correctness": 1,
            "SIAP ID Ambiguity": 0
        }

    def set_default_descriptions(self):
        self.descriptions = {
            "SIAP Completeness": "Fraction of true objects being tracked",
            "SIAP Ambiguity": "Number of tracks assigned to a true object",
            "SIAP Spuriousness": "Fraction of tracks that are unassigned to a true object",
            "SIAP Position Accuracy": "Positional error of associated tracks to their respective "
                                      "truths",
            "SIAP Velocity Accuracy": "Velocity error of associated tracks to their respective "
                                      "truths",
            "SIAP Rate of Track Number Change": "Rate of number of track changes per truth",
            "SIAP Longest Track Segment": "Duration of longest associated track segment per truth",
            "SIAP ID Completeness": "Fraction of true objects with an assigned ID",
            "SIAP ID Correctness": "Fraction of true objects with correct ID assignment",
            "SIAP ID Ambiguity": "Fraction of true objects with ambiguous ID assignment"
        }


class SIAPDiffTableGenerator(SIAPTableGenerator):
    """
    Returns a table displaying the difference between two or more sets of metrics, 
    allowing quick comparison. 
    """
    metrics: list[Collection[MetricGenerator]] = Property(doc="Set of metrics to put in the table")

    metrics_labels: Collection[str] = Property(doc='List of titles for metrics',
                                               default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.targets = None
        self.descriptions = None
        self.set_default_targets()
        self.set_default_descriptions()
        if self.metrics_labels is None:
            self.metrics_labels = ['Metrics ' + str(i) + ' Value'
                                   for i in range(1, len(self.metrics) + 1)]

    def compute_metric(self, **kwargs):
        """Generate table method

        Returns a matplotlib Table of metrics for two sets of values. Table contains metric
        descriptions, target values and a coloured value cell for each set of metrics.
        The colour of each value cell represents how the pair of values of the metric compare to
        each other, with the better value showing in green. Table also contains a 'Diff' value
        displaying the difference between the pair of metric values."""

        white = (1, 1, 1)
        cellText = [["Metric", "Description", "Target"] + self.metrics_labels + ["Max Diff"]]
        cellColors = [[white] * (4 + len(self.metrics))]

        sorted_metrics = [sorted(metric_gens, key=attrgetter('title'))
                          for metric_gens in self.metrics]

        for row_num in range(len(sorted_metrics[0])):
            row_metrics = [m[row_num] for m in sorted_metrics]

            metric_name = row_metrics[0].title
            description = self.descriptions[metric_name]
            target = self.targets[metric_name]

            values = [metric.value for metric in row_metrics]

            diff = round(max(values) - min(values), ndigits=3)

            row_vals = [metric_name, description, target] + \
                       ["{:.2f}".format(value) for value in values] + [diff]

            cellText.append(row_vals)

            colours = []
            for i, value in enumerate(values):
                other_values = values[:i] + values[i+1:]
                if all(num <= 0 for num in [abs(value - target) - abs(v - target)
                                            for v in other_values]):
                    colours.append((0, 1, 0, 0.5))
                elif all(num >= 0 for num in [abs(value - target) - abs(v - target)
                                              for v in other_values]):
                    colours.append((1, 0, 0, 0.5))
                else:
                    colours.append((1, 1, 0, 0.5))

            cellColors.append([white, white, white] + colours + [white])

        # "Plot" table
        scale = (1, 3)
        fig = plt.figure(figsize=(len(cellText)*scale[0] + 1, len(cellText[0])*scale[1]/2))
        ax = fig.add_subplot(1, 1, 1)
        ax.axis('off')
        table = matplotlib.table.table(ax, cellText, cellColors, loc='center')
        table.auto_set_column_width([0, 1, 2, 3])
        table.scale(*scale)

        return table
