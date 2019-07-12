# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines

from .base import PlotGenerator
from ..base import Property
from ..types.detection import Clutter
from ..types.metric import TimeRangePlottingMetric
from ..types.prediction import Prediction
from ..types.time import TimeRange


class TwoDPlotter(PlotGenerator):
    """:class:`~.MetricManager` for the plotting data

    Plots of :class:`~.Track`, :class:`~.Detection` and
    :class:`~.GroundTruthPath` objects in two dimensions.
    """
    track_indices = Property(
        list,
        doc="Elements of track state vector to plot as x and y")
    gtruth_indices = Property(
        list,
        doc="Elements of ground truth path state vector to plot as x and y")
    detection_indices = Property(
        list,
        doc="Elements of detection state vector to plot as x and y")

    def compute_metric(self, manager, *args, **kwargs):
        """Compute the metric using the data in the metric manager

        Parameters
        ----------
        manager : MetricManager
            Containing the data to be used to create the metric(s)

        Returns
        -------
        TimeRangePlottingMetric
            contains a matplotlib figure
        """

        metric = self.plot_tracks_truth_detections(manager.tracks,
                                                   manager.groundtruth_paths,
                                                   manager.detections)
        return metric

    def plot_tracks_truth_detections(self, tracks, groundtruth_paths,
                                     detections):
        """Plots tracks, truths and detections onto a 2d matplotlib figure

        Parameters
        ----------
        tracks: set of :class:`~.Track`
            objects to be plotted as tracks
        groundtruth_paths: set of :class:`~.GroundTruthPath`
            objects to be plotted as truths
        detections: set of :class:`~.Detection`
            objects to be plotted as detections

        Returns
        ----------
        TimeRangePlottingMetric
            Contains the produced plot
        """

        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)

        data = np.array([detection.state_vector for detection in detections
                         if not isinstance(detection, Clutter)])
        if data.any():
            axis.plot(data[:, self.detection_indices[0]],
                      data[:, self.detection_indices[1]],
                      linestyle='', marker='o')

        data = np.array([detection.state_vector for detection in
                         detections if isinstance(detection, Clutter)])
        if data.any():
            axis.plot(data[:, self.detection_indices[0]],
                      data[:, self.detection_indices[1]],
                      linestyle='', marker='2')

        for path in groundtruth_paths:
            data = np.array([state.state_vector for state in path])
            axis.plot(data[:, self.gtruth_indices[0]],
                      data[:, self.gtruth_indices[1]],
                      linestyle=':', marker='')

        for track in tracks:
            if len([state for state in track.states if not isinstance(
                    state, Prediction)]) < 2:
                continue
                # Don't plot tracks with only one detection
                #  associated; probably clutter
            data = np.array([state.state_vector for state in track.states])
            axis.plot(data[:, self.track_indices[0]],
                      data[:, self.track_indices[1]],
                      linestyle='-', marker='.')
            if hasattr(track.state, 'particles'):
                data = np.array(
                    [particle.state_vector for state in track.states for
                     particle in state.particles])
                axis.plot(data[:, self.track_indices[0]],
                          data[:, self.track_indices[1]], linestyle='',
                          marker=".", markersize=1, alpha=0.25)

        axis.set_xlabel("$x$")
        axis.set_ylabel("$y$")
        custom_legend = [
            matplotlib.lines.Line2D([0], [0], color='0', linestyle='',
                                    marker='o'),
            matplotlib.lines.Line2D([0], [0], color='0', linestyle='',
                                    marker='2'),
            matplotlib.lines.Line2D([0], [0], color='0', linestyle=':',
                                    marker=''),
            matplotlib.lines.Line2D([0], [0], color='0', linestyle='-',
                                    marker='.'),
            matplotlib.lines.Line2D([0], [0], color='0', linestyle='',
                                    marker='.', markersize=1),
        ]
        axis.legend(custom_legend,
                    ['Detections', 'Clutter', 'Path', 'Track', 'Particles'])

        timestamps = []
        for state in tracks.union(groundtruth_paths, detections):
            if state.timestamp not in timestamps:
                timestamps.append(state.timestamp)

        return TimeRangePlottingMetric(
            title='Track plot',
            value=fig,
            time_range=TimeRange(min(timestamps), max(timestamps)),
            generator=self)
