# -*- coding: utf-8 -*-
from typing import Tuple

from .base import PlotGenerator
from ..base import Property
from ..types.metric import TimeRangePlottingMetric
from ..types.prediction import Prediction
from ..types.time import TimeRange
from ..plotter import Plotter


class TwoDPlotter(PlotGenerator):
    """:class:`~.MetricManager` for the plotting data

    Plots of :class:`~.Track`, :class:`~.Detection` and
    :class:`~.GroundTruthPath` objects in two dimensions.
    """
    track_indices: Tuple[int, int] = Property(
        doc="Elements of track state vector to plot as x and y")
    gtruth_indices: Tuple[int, int] = Property(
        doc="Elements of ground truth path state vector to plot as x and y")
    detection_indices: Tuple[int, int] = Property(
        doc="Elements of detection state vector to plot as x and y")
    uncertainty: bool = Property(default=False,
                                 doc='If True the plot includes uncertainty ellipses')
    particle: bool = Property(default=False,
                              doc='If True the plot includes particles')

    def compute_metric(self, manager, *args, **kwargs):
        """Compute the metric using the data in the metric manager

        Parameters
        ----------
        manager : MetricManager
            Containing the data to be used to create the metric(s)

        Returns
        -------
        TimeRangePlottingMetric
            Contains a matplotlib figure
        """

        metric = self.plot_tracks_truth_detections(manager.tracks,
                                                   manager.groundtruth_paths,
                                                   manager.detections,
                                                   self.uncertainty,
                                                   self.particle)
        return metric

    def plot_tracks_truth_detections(self, tracks, groundtruth_paths,
                                     detections, uncertainty=False, particle=False):
        """Plots tracks, truths and detections onto a 2d matplotlib figure

        Parameters
        ----------
        tracks: set of :class:`~.Track`
            Objects to be plotted as tracks
        groundtruth_paths: set of :class:`~.GroundTruthPath`
            Objects to be plotted as truths
        detections: set of :class:`~.Detection`
            Objects to be plotted as detections
        uncertainty : bool
            If True, function plots uncertainty ellipses.
        particle : bool
            If True, function plots particles.

        Returns
        ----------
        TimeRangePlottingMetric
            Contains the produced plot
        """

        plotter = Plotter()  # initialises axes using Plotter class

        plotter.plot_measurements(detections, [self.detection_indices[0],
                                               self.detection_indices[1]],
                                  color='tab:blue')

        if groundtruth_paths:
            plotter.plot_ground_truths(groundtruth_paths, [self.gtruth_indices[0],
                                                           self.gtruth_indices[1]],
                                       linestyle=':')
        else:
            groundtruth_paths = []

        if tracks:
            plotting_tracks = set()
            for track in tracks:
                if len([state for state in track.states if not isinstance(
                        state, Prediction)]) >= 2:
                    plotting_tracks.add(track)
                else:
                    continue
            # Don't plot tracks with only one detection associated; probably clutter

            if uncertainty:
                plotter.plot_tracks(plotting_tracks, [self.track_indices[0],
                                                      self.track_indices[1]],
                                    uncertainty=True)

            elif particle:
                plotter.plot_tracks(plotting_tracks, [self.track_indices[0],
                                                      self.track_indices[1]],
                                    particle=True)

            else:
                plotter.plot_tracks(plotting_tracks, [self.track_indices[0],
                                                      self.track_indices[1]])
        else:
            tracks = []

        timestamps = []
        states_list = set()
        for state in states_list.union(tracks, groundtruth_paths, detections):
            if state.timestamp not in timestamps:
                timestamps.append(state.timestamp)

        return TimeRangePlottingMetric(
            title='Track plot',
            value=plotter.fig,
            time_range=TimeRange(min(timestamps), max(timestamps)),
            generator=self)
