import numpy as np
import matplotlib

from .base import MetricGenerator
from ..types import TimePeriodMetric, Metric
from ..types.detection import Clutter

class TwoDPlotter(MetricGenerator):

    def compute_metric(self, tracks, groundtruth_paths, detections, axis = None):

        if not axis:
            axis = matplotlib.pyplot.axes()


        data = np.array([detection.state_vector for detection in detections if not isinstance(detection, Clutter)])
        if data.any():
            axis.plot(data[:, 0], data[:, 1], linestyle='', marker='o')

        data = np.array([detection.state_vector for detection in detections if isinstance(detection, Clutter)])
        if data.any():
            axis.plot(data[:, 0], data[:, 1], linestyle='', marker='2')

        for path in groundtruth_paths:
            data = np.array([state.state_vector for state in path])
            axis.plot(data[:, 0], data[:, 2], linestyle=':', marker='')

        from stonesoup.types.prediction import Prediction
        for track in tracks:
            if len([state for state in track.states if not isinstance(state, Prediction)]) < 2:
                continue  # Don't plot tracks with only one detection associated; probably clutter
            data = np.array([state.state_vector for state in track.states])
            axis.plot(data[:, 0], data[:, 2], linestyle='-', marker='.')
            if hasattr(track.state, 'particles'):
                data = np.array([particle.state_vector for state in track.states for particle in state.particles])
                axis.plot(data[:, 0], data[:, 2], linestyle='', marker=".", markersize=1, alpha=0.25)

        axis.set_xlabel("$x$")
        axis.set_ylabel("$y$")
        custom_legend = [
            matplotlib.lines.Line2D([0], [0], color='0', linestyle='', marker='o'),
            matplotlib.lines.Line2D([0], [0], color='0', linestyle='', marker='2'),
            matplotlib.lines.Line2D([0], [0], color='0', linestyle=':', marker=''),
            matplotlib.lines.Line2D([0], [0], color='0', linestyle='-', marker='.'),
            matplotlib.lines.Line2D([0], [0], color='0', linestyle='', marker='.', markersize=1),
        ]
        axis.legend(custom_legend,
                   ['Detections', 'Clutter', 'Path', 'Track', 'Particles'])

        return TimePeriodMetric(title = 'Track plot',
                            value=axis,
                            start_timestamp=None,
                            end_timestamp=None,
                            generator=self)
