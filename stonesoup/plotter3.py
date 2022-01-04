# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 11:33:27 2021

@author: Peter
"""
import warnings
from itertools import chain

from mpl_toolkits import mplot3d # new as of plotter3

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.legend_handler import HandlerPatch

from .types import detection
from .models.base import LinearModel, NonLinearModel


class Plotter:
    """Plotting class for building graphs of Stone Soup simulations

    A plotting class which is used to simplify the process of plotting ground truths,
    measurements, clutter and tracks. Tracks can be plotted with uncertainty ellipses or
    particles if required. Legends are automatically generated with each plot.

    Attributes
    ----------
    fig: matplotlib.figure.Figure
        Generated figure for graphs to be plotted on
    ax: matplotlib.axes.Axes
        Generated axes for graphs to be plotted on
    handles_list: list of :class:`matplotlib.legend_handler.HandlerBase`
        A list of generated legend handles
    labels_list: list of str
        A list of generated legend labels
    """

    def __init__(self):
        # Generate plot axes
        self.fig = plt.figure(figsize=(10, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel("$x$")
        self.ax.set_ylabel("$y$")
        self.ax.set_zlabel("$z$")
        self.ax.axis('auto')

        # Create empty dictionary for legend handles and labels - dict used to
        # prevent multiple entries with the same label from displaying on legend
        self.legend_dict = {}  # create an empty dictionary to hold legend entries

    def plot_ground_truths(self, truths, mapping, truths_label="Ground Truth", **kwargs):
        """Plots ground truth(s)

        Plots each ground truth path passed in to :attr:`truths` and generates a legend
        automatically. Ground truths are plotted as dashed lines with default colors.

        Users can change linestyle, color and marker using keyword arguments. Any changes
        will apply to all ground truths.

        Parameters
        ----------
        truths : set of :class:`~.GroundTruthPath`
            Set of  ground truths which will be plotted. If not a set, and instead a single
            :class:`~.GroundTruthPath` type, the argument is modified to be a set to allow for
            iteration.
        mapping: list
            List of 2 items specifying the mapping of the x and y components of the state space.
        \\*\\*kwargs: dict
            Additional arguments to be passed to plot function. Default is ``linestyle="--"``.
        """

        truths_kwargs = dict(linestyle="--")
        truths_kwargs.update(kwargs)
        if not isinstance(truths, set):
            truths = {truths}  # Make a set of length 1

        for truth in truths:
            self.ax.plot3D([state.state_vector[mapping[0]] for state in truth],
                         [state.state_vector[mapping[1]] for state in truth],
                         [state.state_vector[mapping[2]] for state in truth],
                         **truths_kwargs)

        # Generate legend items
        truths_handle = Line2D([], [], linestyle=truths_kwargs['linestyle'], color='black')
        self.legend_dict[truths_label] = truths_handle
        # Generate legend
        self.ax.legend(handles=self.legend_dict.values(), labels=self.legend_dict.keys())

    def plot_measurements(self, measurements, mapping, measurement_model=None,
                          measurements_label="Measurements", **kwargs):
        """Plots measurements

        Plots detections and clutter, generating a legend automatically. Detections are plotted as
        blue circles by default unless the detection type is clutter.
        If the detection type is :class:`~.Clutter` it is plotted as a yellow 'tri-up' marker.

        Users can change the color and marker of detections using keyword arguments but not for
        clutter detections.

        Parameters
        ----------
        measurements : list of :class:`~.Detection`
            Detections which will be plotted. If measurements is a set of lists it is flattened.
        mapping: list
            List of 2 items specifying the mapping of the x and y components of the state space.
        measurement_model : :class:`~.Model`, optional
            User-defined measurement model to be used in finding measurement state inverses if
            they cannot be found from the measurements themselves.
        \\*\\*kwargs: dict
            Additional arguments to be passed to plot function for detections. Defaults are
            ``marker='o'`` and ``color='b'``.
        """

        measurement_kwargs = dict(marker='o', color='b')
        measurement_kwargs.update(kwargs)

        if any(isinstance(item, set) for item in measurements):
            measurements_set = chain.from_iterable(measurements)  # Flatten into one set
        else:
            measurements_set = measurements

        plot_detections = []
        plot_clutter = []

        for state in measurements_set:
            meas_model = state.measurement_model  # measurement_model from detections
            if meas_model is None:
                meas_model = measurement_model  # measurement_model from input

            if isinstance(meas_model, LinearModel):  # check to see if meas_model is a LinearModel type
                model_matrix = meas_model.matrix()
                inv_model_matrix = np.linalg.pinv(model_matrix)
                state_vec = inv_model_matrix @ state.state_vector

            elif isinstance(meas_model, NonLinearModel):
                try:
                    state_vec = meas_model.inverse_function(state)
                except (NotImplementedError, AttributeError):
                    warnings.warn('Nonlinear measurement model used with no inverse '
                                  'function available')
                    continue
            else:
                warnings.warn('Measurement model type not specified for all detections')
                continue

            if isinstance(state, detection.Clutter):
                # Plot clutter
                plot_clutter.append((*state_vec[mapping], ))

            elif isinstance(state, detection.Detection):
                # Plot detections
                plot_detections.append((*state_vec[mapping], ))
            else:
                warnings.warn(f'Unknown type {type(state)}')
                continue

        if plot_detections:
            detection_array = np.array(plot_detections)
            self.ax.scatter(detection_array[:, 0], detection_array[:, 1], detection_array[:, 2], **measurement_kwargs)
            measurements_handle = Line2D([], [], linestyle='', **measurement_kwargs)

            # Generate legend items for measurements
            self.legend_dict[measurements_label] = measurements_handle

        if plot_clutter:
            clutter_array = np.array(plot_clutter)
            self.ax.scatter(clutter_array[:, 0], clutter_array[:, 1], clutter_array[:, 2], color='y', marker='2')
            clutter_handle = Line2D([], [], linestyle='', marker='2', color='y')
            clutter_label = "Clutter"

            # Generate legend items for clutter
            self.legend_dict[clutter_label] = clutter_handle

        # Generate legend
        self.ax.legend(handles=self.legend_dict.values(), labels=self.legend_dict.keys())

    def plot_tracks(self, tracks, mapping, uncertainty=False, particle=False, track_label="Track", err_freq=1, 
                    **kwargs):
        """Plots track(s)

        Plots each track generated, generating a legend automatically. If ``uncertainty=True``,
        uncertainty ellipses are plotted. If ``particle=True``, particles are plotted.
        Tracks are plotted as solid lines with point markers and default colors.
        Uncertainty ellipses are plotted with a default color which is the same for all tracks.

        Users can change linestyle, color and marker using keyword arguments. Uncertainty ellipses
        will also be plotted with the user defined colour and any changes will apply to all tracks.

        Parameters
        ----------
        tracks : set of :class:`~.Track`
            Set of tracks which will be plotted. If not a set, and instead a single
            :class:`~.Track` type, the argument is modified to be a set to allow for iteration.
        mapping: list
            List of 2 items specifying the mapping of the x and y components of the state space.
        uncertainty : bool
            If True, function plots uncertainty ellipses.
        particle : bool
            If True, function plots particles.
        track_label: str
            Label to apply to all tracks for legend.
        \\*\\*kwargs: dict
            Additional arguments to be passed to plot function. Defaults are ``linestyle="-"``,
            ``marker='.'`` and ``color=None``.
        """

        tracks_kwargs = dict(linestyle='-', marker=".", color=None)
        tracks_kwargs.update(kwargs)
        if not isinstance(tracks, set):
            tracks = {tracks}  # Make a set of length 1

        # Plot tracks
        track_colors = {}
        for track in tracks:
            line = self.ax.plot([state.state_vector[mapping[0]] for state in track],
                                [state.state_vector[mapping[1]] for state in track],
                                [state.state_vector[mapping[2]] for state in track],
                                **tracks_kwargs)
            track_colors[track] = plt.getp(line[0], 'color')

        # Assuming a single track or all plotted as the same colour then the following will work.
        # Otherwise will just render the final track colour.
        tracks_kwargs['color'] = plt.getp(line[0], 'color')

        # Generate legend items for track
        track_handle = Line2D([], [], linestyle=tracks_kwargs['linestyle'],
                              marker=tracks_kwargs['marker'], color=tracks_kwargs['color'])
        self.legend_dict[track_label] = track_handle

        if uncertainty:
            # Plot uncertainty ellipses
            for track in tracks:
                HH = np.eye(track.ndim)[mapping, :]  # Get position mapping matrix
                check = err_freq
                for state in track:
                    if not check % err_freq:
                        w, v = np.linalg.eig(HH @ state.covar @ HH.T)
                        
                        xl = state.state_vector[mapping[0]]
                        yl = state.state_vector[mapping[1]]
                        zl = state.state_vector[mapping[2]]
                                                
                        x_err = w[0]
                        y_err = w[1]
                        z_err = w[2]
                        
                        self.ax.plot3D([xl+x_err, xl-x_err], [yl, yl], [zl, zl], marker="_", color=tracks_kwargs['color'])
                        self.ax.plot3D([xl, xl], [yl+y_err, yl-y_err], [zl, zl], marker="_", color=tracks_kwargs['color'])
                        self.ax.plot3D([xl, xl], [yl, yl], [zl+z_err, zl-z_err], marker="_", color=tracks_kwargs['color'])
                    check += 1
            self.ax.legend(handles=self.legend_dict.values(), labels=self.legend_dict.keys())

        elif particle:
            # Plot particles
            for track in tracks:
                for state in track:
                    data = state.particles.state_vector[mapping[:2], :]
                    self.ax.plot(data[0], data[1], linestyle='', marker=".",
                                 markersize=1, alpha=0.5)

            # Generate legend items for particles
            particle_handle = Line2D([], [], linestyle='', color="black", marker='.', markersize=1)
            particle_label = "Particles"
            self.legend_dict[particle_label] = particle_handle

            # Generate legend
            self.ax.legend(handles=self.legend_dict.values(), labels=self.legend_dict.keys())

        else:
            self.ax.legend(handles=self.legend_dict.values(), labels=self.legend_dict.keys())

    # Ellipse legend patch (used in Tutorial 3)
    @staticmethod
    def ellipse_legend(ax, label_list, color_list, **kwargs):
        """Adds an ellipse patch to the legend on the axes. One patch added for each item in
        `label_list` with the corresponding color from `color_list`.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Looks at the plot axes defined
        label_list : list of str
            Takes in list of strings intended to label ellipses in legend
        color_list : list of str
            Takes in list of colors corresponding to string/label
            Must be the same length as label_list
        \\*\\*kwargs: dict
                Additional arguments to be passed to plot function. Default is ``alpha=0.2``.
        """

        ellipse_kwargs = dict(alpha=0.2)
        ellipse_kwargs.update(kwargs)

        legend = ax.legend(handler_map={Ellipse: _HandlerEllipse()})
        handles, labels = ax.get_legend_handles_labels()
        for color in color_list:
            handle = Ellipse((0.5, 0.5), 0.5, 0.5, color=color, **ellipse_kwargs)
            handles.append(handle)
        for label in label_list:
            labels.append(label)
        legend._legend_box = None
        legend._init_legend_box(handles, labels)
        legend._set_loc(legend._loc)
        legend.set_title(legend.get_title().get_text())


class _HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5*width - 0.5*xdescent, 0.5*height - 0.5*ydescent
        p = Ellipse(xy=center, width=width + xdescent,
                    height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]
