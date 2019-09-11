import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from .base import MetricTableGenerator
from ..base import Property


class RedGreenTableGenerator(MetricTableGenerator):
    """Red Green Metric Table Generator class

    Takes in a set of metrics and uses known ranges and target
    values to color code an output table with respect to how
    well the tracker performed in relation to each metric, where
    red is worse and green is better"""

    metrics = Property(set, doc="Set of metrics to put in the table")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ranges = dict
        self.targets = dict
        self.descriptions = dict

    def generate_table(self, **kwargs):
        """Generate table method

        Returns a matplotlib Table of metrics with their descriptions, target
        values and a coloured value cell to represent how well the tracker has
        performed in relation to each specific metric (red=bad, green=good)"""

        white = (1, 1, 1)
        cellText = [["Metric", "Description", "Target", "Value"]]
        cellColors = [[white, white, white, white]]

        for metric in self.metrics:
            #  Add metric details to table row
            metric_name = metric.title
            description = self.descriptions[metric_name]
            target = self.targets[metric_name]
            value = metric.value
            cellText.append([metric_name, description, target, value])

            # Generate color for value cell based on closeness to target value
            # Closeness to infinity cannot be represented as a color
            if not target == np.inf:
                red_value = 1
                green_value = 1
                # A value of 1 for both red & green produces yellow

                metric_range = \
                    self.ranges[metric_name][1] - self.ranges[metric_name][0]
                closeness = abs(value - self.targets[metric_name]) \
                    * (1 / metric_range)

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
        fig = plt.figure(figsize=(20, 1))
        ax = fig.add_subplot(1, 1, 1)
        plt.axis("off")
        table = matplotlib.table.table(ax, cellText, cellColors)
        table.auto_set_column_width([0, 1, 2, 3])
        table.scale(1, 4)

        return table


class SIAPTableGenerator(RedGreenTableGenerator):
    """Red Green Table Generator specifically for SIAP metrics

    Contains methods to specify the ranges, descriptions and target values
    specifically for SIAP metrics"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_default_ranges()
        self.set_default_targets()
        self.set_default_descriptions()

    def set_default_ranges(self):
        self.ranges = {
            "SIAP C": (0, 1),
            "SIAP A": (0, 10),  # True maximum is infinity
                                # everything >10 will be treated as "worst"
            "SIAP S": (0, 1),
            "SIAP LT": (0, np.inf),
            "SIAP LS": (0, 1)
        }

    def set_default_targets(self):
        self.targets = {
            "SIAP C": 1,
            "SIAP A": 1,
            "SIAP S": 0,
            "SIAP LT": np.inf,
            "SIAP LS": 1
        }

    def set_default_descriptions(self):
        self.descriptions = {
            "SIAP C": "Completeness, the percentage of live objects with "
                      "tracks on them",
            "SIAP A": "Ambiguity, a measure of the number of tracks assigned "
                      "to each true object",
            "SIAP S": "Spuriousness, the percentage of tracks unassigned to "
                      "any object",
            "SIAP LT": "1/R where R is the average number of excess tracks "
                       "assigned, the higher this value the better",
            "SIAP LS": "The percentage of time spent tracking true objects "
                       "across the dataset"
        }
