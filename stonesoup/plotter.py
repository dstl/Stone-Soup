import warnings
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from itertools import chain
from typing import Collection, Iterable, Union, List, Optional, Tuple, Dict
import numpy as np
from matplotlib import animation as animation
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import kde
try:
    from plotly import colors
except ImportError:
    colors = None
try:
    import plotly.graph_objects as go
except ImportError:
    go = None

from .types import detection
from .types.groundtruth import GroundTruthPath
from .types.array import StateVector
from .types.state import State, StateMutableSequence
from .types.update import Update

from .base import Base, Property

from .models.base import LinearModel, Model

from enum import IntEnum


class Dimension(IntEnum):
    """Dimension Enum class for specifying plotting parameters in the Plotter class.
    Used to sanitize inputs for the dimension attribute of Plotter().

    Attributes
    ----------
    TWO: int
        Specifies 2D plotting for Plotter object
    THREE: int
        Specifies 3D plotting for Plotter object
    """
    ONE = 1  # 1D plotting mode (plot state over time in Plotterly)
    TWO = 2  # 2D plotting mode (original plotter.py functionality)
    THREE = 3  # 3D plotting mode


class _Plotter(ABC):

    @abstractmethod
    def plot_ground_truths(self, truths, mapping, truths_label="Ground Truth", **kwargs):
        raise NotImplementedError

    @abstractmethod
    def plot_measurements(self, measurements, mapping, measurement_model=None,
                          measurements_label="Measurements", **kwargs):
        raise NotImplementedError

    @abstractmethod
    def plot_tracks(self, tracks, mapping, uncertainty=False, particle=False, track_label="Tracks",
                    **kwargs):
        raise NotImplementedError

    @abstractmethod
    def plot_sensors(self, sensors, mapping, sensor_label="Sensors", **kwargs):
        raise NotImplementedError

    def _conv_measurements(self, measurements, mapping, measurement_model=None,
                           convert_measurements=True) -> \
            Tuple[Dict[detection.Detection, StateVector], Dict[detection.Clutter, StateVector]]:
        conv_detections = {}
        conv_clutter = {}
        for state in measurements:
            meas_model = state.measurement_model  # measurement_model from detections
            if meas_model is None:
                meas_model = measurement_model  # measurement_model from input

            if not convert_measurements:
                state_vec = state.state_vector[mapping, :]
            elif isinstance(meas_model, LinearModel):
                model_matrix = meas_model.matrix()
                inv_model_matrix = np.linalg.pinv(model_matrix)
                state_vec = (inv_model_matrix @ state.state_vector)[mapping, :]
            elif isinstance(meas_model, Model):
                try:
                    state_vec = meas_model.inverse_function(state)[mapping, :]
                except (NotImplementedError, AttributeError):
                    warnings.warn('Nonlinear measurement model used with no inverse '
                                  'function available')
                    continue
            else:
                warnings.warn('Measurement model type not specified for all detections')
                continue

            if isinstance(state, detection.Clutter):
                # Plot clutter
                conv_clutter[state] = (*state_vec, )

            elif isinstance(state, detection.Detection):
                # Plot detections
                conv_detections[state] = (*state_vec, )
            else:
                warnings.warn(f'Unknown type {type(state)}')
                continue
        return conv_detections, conv_clutter


class Plotter(_Plotter):
    """Plotting class for building graphs of Stone Soup simulations using matplotlib

    A plotting class which is used to simplify the process of plotting ground truths,
    measurements, clutter and tracks. Tracks can be plotted with uncertainty ellipses or
    particles if required. Legends are automatically generated with each plot.
    Three dimensional plots can be created using the optional dimension parameter.

    Parameters
    ----------
    dimension: enum \'Dimension\'
        Optional parameter to specify 2D or 3D plotting. Default is 2D plotting.
    plot_timeseries: bool
        Specify whether data to be plotted is time series data. Default False
    \\*\\*kwargs: dict
        Additional arguments to be passed to plot function. For example, figsize (Default is
        (10, 6)).

    Attributes
    ----------
    fig: matplotlib.figure.Figure
        Generated figure for graphs to be plotted on
    ax: matplotlib.axes.Axes
        Generated axes for graphs to be plotted on
    legend_dict: dict
        Dictionary of legend handles as :class:`matplotlib.legend_handler.HandlerBase`
        and labels as str
    """

    def __init__(self, dimension=Dimension.TWO, **kwargs):
        figure_kwargs = {"figsize": (10, 6)}
        figure_kwargs.update(kwargs)
        if isinstance(dimension, type(Dimension.TWO)):
            self.dimension = dimension
        elif isinstance(dimension, int):
            self.dimension = Dimension(dimension)
        else:
            raise TypeError("%s is an unsupported type for \'dimension\'; "
                            "expected type %s" % (type(dimension), type(Dimension.TWO)))
        # Generate plot axes
        self.fig = plt.figure(**figure_kwargs)
        if self.dimension is Dimension.TWO:  # 2D axes
            self.ax = self.fig.add_subplot(1, 1, 1)
            self.ax.axis('equal')
        else:  # 3D axes
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.axis('auto')
            self.ax.set_zlabel("$z$")
        self.ax.set_xlabel("$x$")
        self.ax.set_ylabel("$y$")

        # Create empty dictionary for legend handles and labels - dict used to
        # prevent multiple entries with the same label from displaying on legend
        # This is new compared to plotter.py
        self.legend_dict = {}  # create an empty dictionary to hold legend entries

    def plot_ground_truths(self, truths, mapping, truths_label="Ground Truth", **kwargs):
        """Plots ground truth(s)

        Plots each ground truth path passed in to :attr:`truths` and generates a legend
        automatically. Ground truths are plotted as dashed lines with default colors.

        Users can change linestyle, color and marker using keyword arguments. Any changes
        will apply to all ground truths.

        Parameters
        ----------
        truths : Collection of :class:`~.GroundTruthPath`
            Collection of  ground truths which will be plotted. If not a collection and instead a
            single :class:`~.GroundTruthPath` type, the argument is modified to be a set to allow
            for iteration.
        mapping: list
            List of items specifying the mapping of the position components of the state space.
        truths_label: str
            Label for truth data. Default is "Ground Truth"
        \\*\\*kwargs: dict
            Additional arguments to be passed to plot function. Default is ``linestyle="--"``.

        Returns
        -------
        : list of :class:`matplotlib.artist.Artist`
            List of artists that have been added to the axis.
        """
        truths_kwargs = dict(linestyle="--")
        truths_kwargs.update(kwargs)
        if not isinstance(truths, Collection) or isinstance(truths, StateMutableSequence):
            truths = {truths}  # Make a set of length 1

        artists = []
        for truth in truths:
            if self.dimension is Dimension.TWO:  # plots the ground truths in xy
                artists.extend(
                    self.ax.plot([state.state_vector[mapping[0]] for state in truth],
                                 [state.state_vector[mapping[1]] for state in truth],
                                 **truths_kwargs))
            elif self.dimension is Dimension.THREE:  # plots the ground truths in xyz
                artists.extend(
                    self.ax.plot3D([state.state_vector[mapping[0]] for state in truth],
                                   [state.state_vector[mapping[1]] for state in truth],
                                   [state.state_vector[mapping[2]] for state in truth],
                                   **truths_kwargs))
            else:
                raise NotImplementedError('Unsupported dimension type for truth plotting')
        # Generate legend items
        truths_handle = Line2D([], [], linestyle=truths_kwargs['linestyle'], color='black')
        self.legend_dict[truths_label] = truths_handle
        # Generate legend
        artists.append(self.ax.legend(handles=self.legend_dict.values(),
                                      labels=self.legend_dict.keys()))
        return artists

    def plot_measurements(self, measurements, mapping, measurement_model=None,
                          measurements_label="Measurements", convert_measurements=True, **kwargs):
        """Plots measurements

        Plots detections and clutter, generating a legend automatically. Detections are plotted as
        blue circles by default unless the detection type is clutter.
        If the detection type is :class:`~.Clutter` it is plotted as a yellow 'tri-up' marker.

        Users can change the color and marker of detections using keyword arguments but not for
        clutter detections.

        Parameters
        ----------
        measurements : Collection of :class:`~.Detection`
            Detections which will be plotted. If measurements is a set of lists it is flattened.
        mapping: list
            List of items specifying the mapping of the position components of the state space.
        measurement_model : :class:`~.Model`, optional
            User-defined measurement model to be used in finding measurement state inverses if
            they cannot be found from the measurements themselves.
        measurements_label : str
            Label for the measurements.  Default is "Measurements".
        convert_measurements : bool
            Should the measurements be converted from measurement space to state space before
            being plotted. Default is True
        \\*\\*kwargs: dict
            Additional arguments to be passed to plot function for detections. Defaults are
            ``marker='o'`` and ``color='b'``.

        Returns
        -------
        : list of :class:`matplotlib.artist.Artist`
            List of artists that have been added to the axis.
        """

        measurement_kwargs = dict(marker='o', color='b')
        measurement_kwargs.update(kwargs)

        if not isinstance(measurements, Collection):
            measurements = {measurements}  # Make a set of length 1

        if any(isinstance(item, set) for item in measurements):
            measurements_set = chain.from_iterable(measurements)  # Flatten into one set
        else:
            measurements_set = measurements

        plot_detections, plot_clutter = self._conv_measurements(measurements_set,
                                                                mapping,
                                                                measurement_model,
                                                                convert_measurements)

        artists = []
        if plot_detections:
            detection_array = np.array(list(plot_detections.values()))
            # *detection_array.T unpacks detection_array by columns
            # (same as passing in detection_array[:,0], detection_array[:,1], etc...)
            artists.append(self.ax.scatter(*detection_array.T, **measurement_kwargs))
            measurements_handle = Line2D([], [], linestyle='', **measurement_kwargs)

            # Generate legend items for measurements
            self.legend_dict[measurements_label] = measurements_handle

        if plot_clutter:
            clutter_array = np.array(list(plot_clutter.values()))
            artists.append(self.ax.scatter(*clutter_array.T, color='y', marker='2'))
            clutter_handle = Line2D([], [], linestyle='', marker='2', color='y')
            clutter_label = "Clutter"

            # Generate legend items for clutter
            self.legend_dict[clutter_label] = clutter_handle

        # Generate legend
        artists.append(self.ax.legend(handles=self.legend_dict.values(),
                                      labels=self.legend_dict.keys()))
        return artists

    def plot_tracks(self, tracks, mapping, uncertainty=False, particle=False, track_label="Tracks",
                    err_freq=1, same_color=False, **kwargs):
        """Plots track(s)

        Plots each track generated, generating a legend automatically. If ``uncertainty=True``
        and is being plotted in 2D, error ellipses are plotted. If being plotted in
        3D, uncertainty bars are plotted every :attr:`err_freq` measurement, default
        plots uncertainty bars at every track step. Tracks are plotted as solid
        lines with point markers and default colors. Uncertainty bars are plotted
        with a default color which is the same for all tracks.

        Users can change linestyle, color and marker using keyword arguments. Uncertainty metrics
        will also be plotted with the user defined colour and any changes will apply to all tracks.

        Parameters
        ----------
        tracks : Collection of :class:`~.Track`
            Collection of tracks which will be plotted. If not a collection, and instead a single
            :class:`~.Track` type, the argument is modified to be a set to allow for iteration.
        mapping: list
            List of items specifying the mapping of the position
            components of the state space.
        uncertainty : bool
            If True, function plots uncertainty ellipses or bars.
        particle : bool
            If True, function plots particles.
        track_label: str
            Label to apply to all tracks for legend.
        err_freq: int
            Frequency of error bar plotting on tracks. Default value is 1, meaning
            error bars are plotted at every track step.
        same_color: bool
            Should all the tracks have the same color. Default False
        \\*\\*kwargs: dict
            Additional arguments to be passed to plot function. Defaults are ``linestyle="-"``,
            ``marker='s'`` for :class:`~.Update` and ``marker='o'`` for other states.

        Returns
        -------
        : list of :class:`matplotlib.artist.Artist`
            List of artists that have been added to the axis.
        """

        tracks_kwargs = dict(linestyle='-', marker="s", color=None)
        tracks_kwargs.update(kwargs)
        if not isinstance(tracks, Collection) or isinstance(tracks, StateMutableSequence):
            tracks = {tracks}  # Make a set of length 1

        # Plot tracks
        artists = []
        track_colors = {}
        for track in tracks:
            # Get indexes for Update and non-Update states for styling markers
            update_indexes = []
            not_update_indexes = []
            for n, state in enumerate(track):
                if isinstance(state, Update):
                    update_indexes.append(n)
                else:
                    not_update_indexes.append(n)

            data = np.concatenate(
                [(getattr(state, 'mean', state.state_vector)[mapping, :])
                 for state in track],
                axis=1)

            line = self.ax.plot(
                *data,
                markevery=update_indexes,
                **tracks_kwargs)
            artists.extend(line)
            if not_update_indexes:
                artists.extend(self.ax.plot(
                    *data[:, not_update_indexes],
                    marker="o" if "marker" not in kwargs else kwargs['marker'],
                    linestyle='',
                    color=plt.getp(line[0], 'color')))
            track_colors[track] = plt.getp(line[0], 'color')
            if same_color:
                tracks_kwargs['color'] = plt.getp(line[0], 'color')

        if tracks:  # If no tracks `line` won't be defined
            # Assuming a single track or all plotted as the same colour then the following will
            # work. Otherwise will just render the final track colour.
            tracks_kwargs['color'] = plt.getp(line[0], 'color')

        # Generate legend items for track
        track_handle = Line2D([], [], linestyle=tracks_kwargs['linestyle'],
                              marker=tracks_kwargs['marker'], color=tracks_kwargs['color'])
        self.legend_dict[track_label] = track_handle
        if uncertainty:
            if self.dimension is Dimension.TWO:
                # Plot uncertainty ellipses
                for track in tracks:
                    HH = np.eye(track.ndim)[mapping, :]  # Get position mapping matrix
                    check = err_freq - 1    # plot the first one
                    for state in track:
                        check += 1
                        if check % err_freq:
                            continue
                        w, v = np.linalg.eig(HH @ state.covar @ HH.T)
                        if np.iscomplexobj(w) or np.iscomplexobj(v):
                            warnings.warn("Can not plot uncertainty for all states due to complex "
                                          "eignevalues or eigenvectors", UserWarning)
                            continue
                        max_ind = np.argmax(w)
                        min_ind = np.argmin(w)
                        orient = np.arctan2(v[1, max_ind], v[0, max_ind])
                        ellipse = Ellipse(xy=state.mean[mapping[:2], 0],
                                          width=2 * np.sqrt(w[max_ind]),
                                          height=2 * np.sqrt(w[min_ind]),
                                          angle=np.rad2deg(orient), alpha=0.2,
                                          color=track_colors[track])
                        self.ax.add_artist(ellipse)
                        artists.append(ellipse)

                # Generate legend items for uncertainty ellipses
                ellipse_handle = Ellipse((0.5, 0.5), 0.5, 0.5, alpha=0.2,
                                         color=tracks_kwargs['color'])
                ellipse_label = "Uncertainty"
                self.legend_dict[ellipse_label] = ellipse_handle
                # Generate legend
                artists.append(self.ax.legend(handles=self.legend_dict.values(),
                                              labels=self.legend_dict.keys(),
                                              handler_map={Ellipse: _HandlerEllipse()}))
            else:
                # Plot 3D error bars on tracks
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

                            artists.extend(
                                self.ax.plot3D([xl+x_err, xl-x_err], [yl, yl], [zl, zl],
                                               marker="_", color=tracks_kwargs['color']))
                            artists.extend(
                                self.ax.plot3D([xl, xl], [yl+y_err, yl-y_err], [zl, zl],
                                               marker="_", color=tracks_kwargs['color']))
                            artists.extend(
                                self.ax.plot3D([xl, xl], [yl, yl], [zl+z_err, zl-z_err],
                                               marker="_", color=tracks_kwargs['color']))
                        check += 1

        if particle:
            if self.dimension is Dimension.TWO:
                # Plot particles
                for track in tracks:
                    for state in track:
                        data = state.state_vector[mapping[:2], :]
                        artists.extend(self.ax.plot(data[0], data[1], linestyle='', marker=".",
                                                    markersize=1, alpha=0.5))

                # Generate legend items for particles
                particle_handle = Line2D([], [], linestyle='', color="black", marker='.',
                                         markersize=1)
                particle_label = "Particles"
                self.legend_dict[particle_label] = particle_handle
                # Generate legend
                artists.append(self.ax.legend(handles=self.legend_dict.values(),
                                              labels=self.legend_dict.keys()))
            else:
                raise NotImplementedError("""Particle plotting is not currently supported for
                                          3D visualization""")

        else:
            artists.append(self.ax.legend(handles=self.legend_dict.values(),
                                          labels=self.legend_dict.keys()))

        return artists

    def plot_sensors(self, sensors, mapping=None, sensor_label="Sensors", **kwargs):
        """Plots sensor(s)

        Plots sensors.  Users can change the color and marker of sensors using keyword
        arguments. Default is a black 'x' marker.

        Parameters
        ----------
        sensors : Collection of :class:`~.Sensor`
            Sensors to plot
        mapping: list
            List of items specifying the mapping of the position components of the
            sensor's position. Default is either [0, 1] or [0, 1, 2] depending on `self.dimension`
        sensor_label: str
            Label to apply to all sensors for legend.
        \\*\\*kwargs: dict
            Additional arguments to be passed to plot function for sensors. Defaults are
            ``marker='x'`` and ``color='black'``.

        Returns
        -------
        : list of :class:`matplotlib.artist.Artist`
            List of artists that have been added to the axis.
        """

        sensor_kwargs = dict(marker='x', color='black')
        sensor_kwargs.update(kwargs)

        if not isinstance(sensors, Collection):
            sensors = {sensors}  # Make a set of length 1

        if mapping is None:
            mapping = list(range(self.dimension))

        artists = []
        for sensor in sensors:
            if self.dimension is Dimension.TWO:  # plots the sensors in xy
                artists.append(self.ax.scatter(sensor.position[mapping[0]],
                                               sensor.position[mapping[1]],
                                               **sensor_kwargs))
            elif self.dimension is Dimension.THREE:  # plots the sensors in xyz
                artists.extend(self.ax.plot3D(sensor.position[mapping[0]],
                                              sensor.position[mapping[1]],
                                              sensor.position[mapping[2]],
                                              **sensor_kwargs))
            else:
                raise NotImplementedError('Unsupported dimension type for sensor plotting')
        self.legend_dict[sensor_label] = Line2D([], [], linestyle='', **sensor_kwargs)
        artists.append(self.ax.legend(handles=self.legend_dict.values(),
                                      labels=self.legend_dict.keys()))
        return artists

    def set_equal_3daxis(self, axes=None):
        """Plots minimum/maximum points with no linestyle to increase the plotting region to
        simulate `.ax.axis('equal')` from matplotlib 2d plots which is not possible using 3d
        projection.

        Parameters
        ----------
        axes: list
            List of dimension index specifying the equal axes, equal x and y = [0,1].
            Default is x,y [0,1].
        """
        if not axes:
            axes = [0, 1]
        if self.dimension is Dimension.THREE:
            min_xyz = [0, 0, 0]
            max_xyz = [0, 0, 0]
            for n in range(3):
                for line in self.ax.lines:
                    min_xyz[n] = np.min([min_xyz[n], *line.get_data_3d()[n]])
                    max_xyz[n] = np.max([max_xyz[n], *line.get_data_3d()[n]])

            extremes = np.max([x - y for x, y in zip(max_xyz, min_xyz)])
            equal_axes = [0, 0, 0]
            for i in axes:
                equal_axes[i] = 1
            lower = ([np.mean([x, y]) for x, y in zip(max_xyz, min_xyz)] - extremes/2) * equal_axes
            upper = ([np.mean([x, y]) for x, y in zip(max_xyz, min_xyz)] + extremes/2) * equal_axes
            ghosts = GroundTruthPath(states=[State(state_vector=lower),
                                             State(state_vector=upper)])

            self.ax.plot3D([state.state_vector[0] for state in ghosts],
                           [state.state_vector[1] for state in ghosts],
                           [state.state_vector[2] for state in ghosts],
                           linestyle="")

    def plot_density(self, state_sequences: Collection[StateMutableSequence],
                     index: Union[int, None] = -1,
                     mapping=(0, 2), n_bins=300, **kwargs):
        """

        Parameters
        ----------
        state_sequences : a collection of :class:`~.StateMutableSequence`
            Set of tracks which will be plotted. If not a set, and instead a single
            :class:`~.Track` type, the argument is modified to be a set to allow for iteration.
        index: int
            Which index of the StateMutableSequences should be plotted.
            Default value is '-1' which is the last state in the sequences.
            index can be set to None if all indices of the sequence should be included in the plot
        mapping: list
            List of 2 items specifying the mapping of the x and y components of the state space.
        n_bins : int
            Size of the bins used to group the data
        \\*\\*kwargs: dict
            Additional arguments to be passed to pcolormesh function.
        """
        if len(state_sequences) == 0:
            raise ValueError("Skipping plotting density due to state_sequences being empty.")
        if index is None:  # Plot all states in the sequence
            x = np.array([a_state.state_vector[mapping[0]]
                          for a_state_sequence in state_sequences
                          for a_state in a_state_sequence])
            y = np.array([a_state.state_vector[mapping[1]]
                          for a_state_sequence in state_sequences
                          for a_state in a_state_sequence])
        else:  # Only plot one state out of the sequences
            x = np.array([a_state_sequence.states[index].state_vector[mapping[0]]
                          for a_state_sequence in state_sequences])
            y = np.array([a_state_sequence.states[index].state_vector[mapping[1]]
                          for a_state_sequence in state_sequences])
        if np.allclose(x, y, atol=1e-10):
            raise ValueError("Skipping plotting density due to x and y values are the same. "
                             "This leads to a singular matrix in the kde function.")
        # Evaluate a gaussian kde on a regular grid of n_bins x n_bins over data extents
        k = kde.gaussian_kde([x, y])
        xi, yi = np.mgrid[x.min():x.max():n_bins * 1j, y.min():y.max():n_bins * 1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        # Make the plot
        self.ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', **kwargs)

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


class MetricPlotter(ABC):
    """Class for plotting Stone Soup metrics using matplotlib

    A plotting class which is used to simplify the process of plotting metrics.
    Legends are automatically generated with each plot.

    """
    def __init__(self):
        self.fig = None
        self.axes = None
        self.plottable_metrics = ["OSPA distances",
                                  "GOSPA Metrics",
                                  "SIAP Completeness at times",
                                  "SIAP Ambiguity at times",
                                  "SIAP Spuriousness at times",
                                  "SIAP Position Accuracy at times",
                                  "SIAP Velocity Accuracy at times",
                                  "SIAP ID Completeness at times",
                                  "SIAP ID Correctness at times",
                                  "SIAP ID Ambiguity at times",
                                  "PCRB Metrics",
                                  "Sum of Covariance Norms Metric",
                                  "Mean of Covariance Norms Metric"
                                  ]

    def plot_metrics(self, metrics, generator_names=None, metric_names=None,
                     combine_plots=True, **kwargs):
        """Plots metrics

        Plots each plottable metric passed in to :attr:`metrics` across a series of subplots
        and generates legend(s) automatically. Metrics are plotted as lines with default colors.

        Users can change linestyle, color and marker or other features using keyword arguments.
        Any changes will apply to all metrics.

        Parameters
        ----------
        metrics : dict of :class:`~.Metric`
            Dictionary of generated metrics to be plotted.
        generator_names: list of str
            Generator(s) to extract specific metrics from :attr:`metrics` for plotting.
            Default None to take all metrics.
        metric_names: list of str
            Specific metric(s) to extract from :class:`~.MetricGenerator` for plotting.
            Default None to take all metrics in generators.
        combine_plots: bool
            Plot metrics of same type on the same subplot. Default True.
        \\*\\*kwargs: dict
            Additional arguments to be passed to plot function. Default is ``linestyle="-"``.

        Returns
        -------
        : :class:`matplotlib.pyplot.figure`
            Figure containing subplots displaying all plottable metrics.
        """
        metrics_kwargs = dict(linestyle="-")
        metrics_kwargs.update(kwargs)

        generator_names = list(metrics.keys()) if generator_names is None else generator_names

        # warning for user input metrics that will not be plotted
        if metric_names is not None:
            for metric_name in metric_names:
                if metric_name not in self.plottable_metrics:
                    warnings.warn(f"{metric_name} "
                                  f"is not a plottable metric and will not be plotted")
        else:
            metric_names = self.extract_metric_types(metrics)

        metrics_to_plot = self._extract_plottable_metrics(metrics, generator_names, metric_names)

        if combine_plots:
            self.combine_plots(metrics_to_plot, metrics_kwargs)
        else:
            self.plot_separately(metrics_to_plot, metrics_kwargs)

    def _extract_plottable_metrics(self, metrics, generator_names, metric_names):
        """
        Extract all plottable metrics from dict of generated metrics.

        Parameters
        ----------
        metrics: dict of :class:`~.Metric`
            Dictionary of generated metrics.
        generator_names: list of str
            Generator(s) to extract specific metrics from :attr:`metrics` for plotting.
        metric_names: list of str
            Specific metric(s) to extract from :class:`~.MetricGenerator` for plotting.

        Returns
        -------
        : dict
            Dict of all plottable metrics.
        """
        metrics_dict = dict()

        for generator_name in generator_names:
            for metric_name in metric_names:
                if metric_name in metrics[generator_name].keys() and \
                        metric_name in self.plottable_metrics:
                    if generator_name not in metrics_dict.keys():
                        metrics_dict[generator_name] = \
                            {metric_name: metrics[generator_name][metric_name]}
                    else:
                        metrics_dict[generator_name][metric_name] = \
                            metrics[generator_name][metric_name]

        return metrics_dict

    def _count_subplots(self, metrics_to_plot, combine_plots):
        """
        Calculate number of subplots needed to plot all metrics.

        Parameters
        ----------
        metrics_to_plot: dict of :class:`~.Metric`
            Dictionary of metrics to be plotted.
        combine_plots: bool
            Specifies whether same metric types should be plotted on same subplot.

        Returns
        -------
        : int
            Number of subplots to generate.
        """
        if combine_plots:
            metric_types = self.extract_metric_types(metrics_to_plot)
            number_of_subplots = len(metric_types)

        else:
            number_of_subplots = 0
            for generator in metrics_to_plot.keys():
                number_of_subplots += len(metrics_to_plot[generator])

        return number_of_subplots

    @staticmethod
    def extract_metric_types(metrics):
        """
        Identify the different types of metric held in dict of metrics.

        Parameters
        ----------
        metrics: dict of :class:`~.Metric`
            Dictionary of metrics.

        Returns
        -------
        : list
            Sorted list of types of metric
        """
        metric_types = set()
        for generator in metrics.keys():
            for metric_key in metrics[generator].keys():
                metric_types.add(metric_key)

        metric_types = list(metric_types)
        metric_types.sort()

        return metric_types

    def combine_plots(self, metrics_to_plot, metrics_kwargs):
        """
        Generates one subplot for each different metric type and plots metrics of the same
        type on same subplot. Metrics are plotted over time.

        Parameters
        ----------
        metrics_to_plot: dict of :class:`~.Metric`
            Dictionary of metrics to plot.
        metrics_kwargs: dict
            Keyword arguments to be passed to plot function.

        Returns
        -------
        : :class:`matplotlib.pyplot.figure`
            Figure containing subplots displaying metrics.
        """
        # determine how many plots required - equal to number of metric types
        number_of_subplots = self._count_subplots(metrics_to_plot, True)

        # initialise each subplot
        self.fig, axes = plt.subplots(number_of_subplots, figsize=(10, 6*number_of_subplots))
        self.fig.subplots_adjust(hspace=0.3)

        # extract data for each subplot and plot it
        metric_types = self.extract_metric_types(metrics_to_plot)

        self.axes = axes if isinstance(axes, Iterable) else [axes]

        # generate colour map for lines to be plotted
        if 'color' not in metrics_kwargs.keys():
            colour_map = plt.cm.rainbow(np.linspace(0, 1, len(metrics_to_plot.keys())))
        else:
            colour_map = metrics_kwargs['color']
            metrics_kwargs.pop('color')

        for metric_type, axis in zip(list(metric_types), self.axes):
            artists = []
            legend_dict = {}

            colour_map_copy = iter(colour_map.copy())

            for generator in metrics_to_plot.keys():
                for metric in metrics_to_plot[generator].keys():
                    if metric == metric_type:
                        colour = next(colour_map_copy)
                        metric_values = metrics_to_plot[generator][metric].value
                        artists.extend(axis.plot([_.timestamp for _ in metric_values],
                                                 [_.value for _ in metric_values],
                                                 color=colour,
                                                 **metrics_kwargs))

                        metric_handle = Line2D([], [], linestyle=metrics_kwargs['linestyle'],
                                               color=colour)
                        legend_dict[generator] = metric_handle

            # Generate legend
            artists.append(axis.legend(handles=legend_dict.values(),
                                       labels=legend_dict.keys()))

            y_label = metric_type.split(' at times')[0]
            artists.extend(axis.set(title=metric_type.split(' at times')[0],
                                    xlabel="Time", ylabel=y_label))

    def plot_separately(self, metrics_to_plot, metrics_kwargs):
        """
        Generates one subplot for each different individual metric and plots metric
        values over time.

        Parameters
        ----------
        metrics_to_plot: dict of :class:`~.Metric`
            Dictionary of metrics to plot.
        metrics_kwargs: dict
            Keyword arguments to be passed to plot function.

        Returns
        -------
        : :class:`matplotlib.pyplot.figure`
            Figure containing subplots displaying metrics.
        """
        metrics_kwargs['color'] = metrics_kwargs['color'] if \
            'color' in metrics_kwargs.keys() else 'blue'

        # determine how many plots required - equal to number of metrics within the generators
        number_of_subplots = self._count_subplots(metrics_to_plot, False)

        # initialise each plot
        self.fig, axes = plt.subplots(number_of_subplots, figsize=(10, 6*number_of_subplots))
        self.fig.subplots_adjust(hspace=0.3)

        # extract data for each plot and plot it
        all_metrics = {}
        for generator in metrics_to_plot.keys():
            for metric in list(metrics_to_plot[generator].keys()):
                all_metrics[f'{generator}: {metric}'] = metrics_to_plot[generator][metric]

        self.axes = axes if isinstance(axes, Iterable) else [axes]

        for metric, axis in zip(all_metrics.keys(), self.axes):
            y_label = str(all_metrics[metric].title).split(' at times')[0]
            axis.set(title=str(all_metrics[metric].title), xlabel='Time', ylabel=y_label)
            metric_values = all_metrics[metric].value
            axis.plot([_.timestamp for _ in metric_values],
                      [_.value for _ in metric_values],
                      **metrics_kwargs)

            # Generate legend
            metric_handle = Line2D([], [], linestyle=metrics_kwargs['linestyle'],
                                   color=metrics_kwargs['color'])
            axis.legend(handles=[metric_handle],
                        labels=[metric.split(' at times')[0]])

    def set_fig_title(self, title):
        """
        Set title for the figure.

        Parameters
        ----------
        title: str
            Figure title text.

        Returns
        -------
        Text instance of figure title.
        """
        self.fig.suptitle(t=title)

    def set_ax_title(self, titles):
        """
        Set axis titles for each axis in figure.

        Parameters
        ----------
        titles: list of str
            List of strings for title text for each axis.

        Returns
        -------
        Text instance of axis titles.
        """
        for axis, title in zip(self.axes, titles):
            axis.set(title=title)


class Plotterly(_Plotter):
    """Plotting class for building graphs of Stone Soup simulations using plotly

    A plotting class which is used to simplify the process of plotting ground truths,
    measurements, clutter and tracks. Tracks can be plotted with uncertainty ellipses or
    particles if required. Legends are automatically generated with each plot.
    Three-dimensional plots can be created using the optional dimension parameter.

    Parameters
    ----------
    dimension: enum \'Dimension\'
        Optional parameter to specify 1D, 2D, or 3D plotting.
    \\*\\*kwargs: dict
        Additional arguments to be passed to the Plotly.graph_objects Figure.

    Attributes
    ----------
    fig: plotly.graph_objects.Figure
        Generated figure to display graphs.
    """
    def __init__(self, dimension=Dimension.TWO, **kwargs):
        if go is None:
            raise RuntimeError("Usage of Plotterly plotter requires installation of `plotly`")

        self.dimension = Dimension(dimension)  # allows 1, 2, 3,
        # Dimension(1), Dimension(2) or Dimension(3)

        from plotly import colors
        layout_kwargs = dict(
            xaxis_title="x",
            yaxis_title="y",
            colorway=colors.qualitative.Plotly,  # Needed to match colours later.
        )

        if self.dimension == 3:
            layout_kwargs.update(dict(scene_aspectmode='data'))  # auto shapes fig to fit data well

        layout_kwargs.update(kwargs)

        # Generate plot axes
        self.fig = go.Figure(layout=layout_kwargs)

    @staticmethod
    def _format_state_text(state):
        text = []
        text.append(type(state).__name__)
        text.append(getattr(state, 'mean', state.state_vector))
        text.append(state.timestamp)
        text.extend([f"{key}: {value}" for key, value in getattr(state, 'metadata', {}).items()])

        return "<br>".join((str(t) for t in text))

    def _check_mapping(self, mapping):
        if len(mapping) == 0:
            raise ValueError("No indices provided in mapping.")
        elif len(mapping) != self.dimension:
            raise TypeError("Plotter dimension is not same as the mapping dimension.")

    def plot_ground_truths(self, truths, mapping, truths_label="Ground Truth", **kwargs):
        """Plots ground truth(s)

        Plots each ground truth path passed in to :attr:`truths` and generates a legend
        automatically. Ground truths are plotted as dashed lines with default colors.

        Users can change line style, color and marker using keyword arguments. Any changes
        will apply to all ground truths.

        Parameters
        ----------
        truths : Collection of :class:`~.GroundTruthPath`
            Collection of  ground truths which will be plotted. If not a collection,
            and instead a single :class:`~.GroundTruthPath` type, the argument is modified to be a
            set to allow for iteration.
        mapping: list
            List of items specifying the mapping of the position components of the state space.
        truths_label: str
            Label for truth data. Default is "Ground Truth"
        \\*\\*kwargs: dict
            Additional arguments to be passed to scatter function. Default is
            ``line=dict(dash="dash")``.
        """
        if not isinstance(truths, Collection) or isinstance(truths, StateMutableSequence):
            truths = {truths}

        self._check_mapping(mapping)  # ensure mapping is compatible with plotter dimension

        truths_kwargs = dict(
            mode="lines", line=dict(dash="dash"), legendgroup=truths_label, legendrank=100,
            name=truths_label)

        if self.dimension == 3:  # make ground truth line thicker so easier to see in 3d plot
            truths_kwargs.update(dict(line=dict(width=8, dash="longdashdot")))

        truths_kwargs.update(kwargs)
        add_legend = truths_kwargs['legendgroup'] not in {trace.legendgroup
                                                          for trace in self.fig.data}

        for truth in truths:
            scatter_kwargs = truths_kwargs.copy()
            if add_legend:
                scatter_kwargs['showlegend'] = True
                add_legend = False
            else:
                scatter_kwargs['showlegend'] = False

            if self.dimension == 1:
                self.fig.add_scatter(
                    x=[state.timestamp for state in truth],
                    y=[state.state_vector[mapping[0]] for state in truth],
                    text=[self._format_state_text(state) for state in truth],
                    **scatter_kwargs)

            elif self.dimension == 2:
                self.fig.add_scatter(
                    x=[state.state_vector[mapping[0]] for state in truth],
                    y=[state.state_vector[mapping[1]] for state in truth],
                    text=[self._format_state_text(state) for state in truth],
                    **scatter_kwargs)

            elif self.dimension == 3:
                self.fig.add_scatter3d(
                    x=[state.state_vector[mapping[0]] for state in truth],
                    y=[state.state_vector[mapping[1]] for state in truth],
                    z=[state.state_vector[mapping[2]] for state in truth],
                    text=[self._format_state_text(state) for state in truth],
                    **scatter_kwargs)

    def plot_measurements(self, measurements, mapping, measurement_model=None,
                          measurements_label="Measurements", convert_measurements=True, **kwargs):
        """Plots measurements

        Plots detections and clutter, generating a legend automatically. Detections are plotted as
        blue circles by default unless the detection type is clutter.
        If the detection type is :class:`~.Clutter` it is plotted as a yellow 'tri-up' marker.

        Users can change the color and marker of detections using keyword arguments but not for
        clutter detections.

        Parameters
        ----------
        measurements : Collection of :class:`~.Detection`
            Detections which will be plotted. If measurements is a set of lists it is flattened.
        mapping: list
            List of items specifying the mapping of the position components of the state space.
        measurement_model : :class:`~.Model`, optional
            User-defined measurement model to be used in finding measurement state inverses if
            they cannot be found from the measurements themselves.
        measurements_label : str
            Label for the measurements.  Default is "Measurements".
        convert_measurements: bool
            Should the measurements be converted from measurement space to state space before
            being plotted. Default is True
        \\*\\*kwargs: dict
            Additional arguments to be passed to scatter function for detections. Defaults are
            ``marker=dict(color="#636EFA")``.
        """

        if not isinstance(measurements, Collection):
            measurements = {measurements}

        if any(isinstance(item, set) for item in measurements):
            measurements_set = chain.from_iterable(measurements)  # Flatten into one set
        else:
            measurements_set = set(measurements)

        self._check_mapping(mapping)

        plot_detections, plot_clutter = self._conv_measurements(measurements_set,
                                                                mapping,
                                                                measurement_model,
                                                                convert_measurements)

        if plot_detections:
            name = measurements_label + "<br>(Detections)"
            measurement_kwargs = dict(
                mode='markers', marker=dict(color='#636EFA'),
                name=name, legendgroup=name, legendrank=200)

            if self.dimension == 3:  # make markers smaller in 3d plot
                measurement_kwargs.update(dict(marker=dict(size=4, color='#636EFA')))

            measurement_kwargs.update(kwargs)
            if measurement_kwargs['legendgroup'] not in {trace.legendgroup
                                                         for trace in self.fig.data}:
                measurement_kwargs['showlegend'] = True
            else:
                measurement_kwargs['showlegend'] = False
            detection_array = np.asarray(list(plot_detections.values()), dtype=np.float64)

            if self.dimension == 1:
                self.fig.add_scatter(
                    x=[state.timestamp for state in plot_detections.keys()],
                    y=detection_array[:, 0],
                    text=[self._format_state_text(state) for state in plot_detections.keys()],
                    **measurement_kwargs,
                )
            elif self.dimension == 2:
                self.fig.add_scatter(
                    x=detection_array[:, 0],
                    y=detection_array[:, 1],
                    text=[self._format_state_text(state) for state in plot_detections.keys()],
                    **measurement_kwargs,
                )
            elif self.dimension == 3:
                self.fig.add_scatter3d(
                    x=detection_array[:, 0],
                    y=detection_array[:, 1],
                    z=detection_array[:, 2],
                    text=[self._format_state_text(state) for state in plot_detections.keys()],
                    **measurement_kwargs,
                )

        if plot_clutter:
            name = measurements_label + "<br>(Clutter)"
            measurement_kwargs = dict(
                mode='markers', marker=dict(symbol="star-triangle-up", color='#FECB52'),
                name=name, legendgroup=name, legendrank=210)

            if self.dimension == 3:  # update - star-triangle-up not in 3d plotly
                measurement_kwargs.update(dict(marker=dict(size=4, symbol="diamond",
                                                           color='#FECB52')))

            measurement_kwargs.update(kwargs)
            if measurement_kwargs['legendgroup'] not in {trace.legendgroup
                                                         for trace in self.fig.data}:
                measurement_kwargs['showlegend'] = True
            else:
                measurement_kwargs['showlegend'] = False
            clutter_array = np.asarray(list(plot_clutter.values()), dtype=np.float64)

            if self.dimension == 1:
                self.fig.add_scatter(
                    x=[state.timestamp for state in plot_clutter.keys()],
                    y=clutter_array[:, 0],
                    text=[self._format_state_text(state) for state in plot_clutter.keys()],
                    **measurement_kwargs,
                )
            elif self.dimension == 2:
                self.fig.add_scatter(
                    x=clutter_array[:, 0],
                    y=clutter_array[:, 1],
                    text=[self._format_state_text(state) for state in plot_clutter.keys()],
                    **measurement_kwargs,
                )
            elif self.dimension == 3:
                self.fig.add_scatter3d(
                    x=clutter_array[:, 0],
                    y=clutter_array[:, 1],
                    z=clutter_array[:, 2],
                    text=[self._format_state_text(state) for state in plot_clutter.keys()],
                    **measurement_kwargs,
                )

    def get_next_color(self):
        """
        Find the colour of the next plot. This approach to getting colour isn't ideal, but should
        work in most cases...
        Returns
        -------
        dist : str
            Hex string for a colour
        """
        # Find how many sequences have been plotted so far. The current plot has already been added
        # to fig.data, so -1 is needed
        figure_index = len(self.fig.data) - 1

        # Get the list of colours used for plotting
        colorway = self.fig.layout.colorway
        max_index = len(colorway)

        # Use the modulo operator to limit the colour index to limits of the colorway.
        # If figure_index > max_index then colours will be reused
        color_index = figure_index % max_index
        return colorway[color_index]

    def plot_tracks(self, tracks, mapping, uncertainty=False, particle=False, track_label="Tracks",
                    ellipse_points=30, err_freq=1, same_color=False, **kwargs):
        """Plots track(s)

        Plots each track generated, generating a legend automatically. If ``uncertainty=True``
        error ellipses are plotted.
        Tracks are plotted as solid lines with point markers and default colors.

        Users can change line style, color and marker using keyword arguments.

        Parameters
        ----------
        tracks : Collection of :class:`~.Track`
            Collection of tracks which will be plotted. If not a collection, and instead a single
            :class:`~.Track` type, the argument is modified to be a set to allow for iteration.
        mapping: list
            List of items specifying the mapping of the position
            components of the state space.
        uncertainty : bool
            If True, function plots uncertainty ellipses.
        particle : bool
            If True, function plots particles.
        track_label: str
            Label to apply to all tracks for legend.
        ellipse_points: int
            Number of points for polygon approximating ellipse shape
        err_freq: int
            Frequency of error bar plotting on tracks. Default value is 1, meaning
            error bars are plotted at every track step.
        same_color: bool
            Should all the tracks have the same colour. Default False
        \\*\\*kwargs: dict
            Additional arguments to be passed to scatter function. Defaults are
            ``marker=dict(symbol='square')`` for :class:`~.Update` and
            ``marker=dict(symbol='circle')`` for other states.
        """
        if not isinstance(tracks, Collection) or isinstance(tracks, StateMutableSequence):
            tracks = {tracks}  # Make a set of length 1

        self._check_mapping(mapping)  # check size of mapping against dimension of plotter

        # Plot tracks
        track_colors = {}
        track_kwargs = dict(mode='markers+lines', legendgroup=track_label, legendrank=300)

        if self.dimension == 3:  # change visuals to work well in 3d
            track_kwargs.update(dict(line=dict(width=7)), marker=dict(size=4))
        track_kwargs.update(kwargs)
        add_legend = track_kwargs['legendgroup'] not in {trace.legendgroup
                                                         for trace in self.fig.data}

        if same_color:
            color = track_kwargs.get('marker', {}).get('color') or \
                    track_kwargs.get('line', {}).get('color')

            # Set the colour if it hasn't already been set
            if color is None:
                track_kwargs['marker'] = track_kwargs.get('marker', {})
                track_kwargs['marker']['color'] = self.get_next_color()

        for track in tracks:
            scatter_kwargs = track_kwargs.copy()
            scatter_kwargs['name'] = track.id
            if add_legend:
                scatter_kwargs['name'] = track_label
                scatter_kwargs['showlegend'] = True
                add_legend = False
            else:
                scatter_kwargs['showlegend'] = False
            scatter_kwargs['marker'] = scatter_kwargs.get('marker', {}).copy()
            if 'symbol' not in scatter_kwargs['marker']:
                scatter_kwargs['marker']['symbol'] = [
                    'square' if isinstance(state, Update) else 'circle' for state in track]

            if len(self.fig.data) > 0:
                track_colors[track] = (self.fig.data[-1].line.color
                                       or self.fig.data[-1].marker.color
                                       or self.get_next_color())
            else:
                track_colors[track] = self.get_next_color()

            if self.dimension == 1:  # plot 1D tracks

                if uncertainty or particle:
                    raise NotImplementedError

                self.fig.add_scatter(
                    x=[state.timestamp for state in track],
                    y=[float(getattr(state, 'mean', state.state_vector)[mapping[0]])
                       for state in track],
                    text=[self._format_state_text(state) for state in track],
                    **scatter_kwargs)

            elif self.dimension == 2:  # plot 2D tracks

                self.fig.add_scatter(
                    x=[float(getattr(state, 'mean', state.state_vector)[mapping[0]])
                       for state in track],
                    y=[float(getattr(state, 'mean', state.state_vector)[mapping[1]])
                       for state in track],
                    text=[self._format_state_text(state) for state in track],
                    **scatter_kwargs)

            elif self.dimension == 3:  # plot 3D tracks

                if particle:
                    raise NotImplementedError

                # create empty error arrays
                err_x = np.array([np.nan for _ in range(len(track))], dtype=float)
                err_y = np.array([np.nan for _ in range(len(track))], dtype=float)
                err_z = np.array([np.nan for _ in range(len(track))], dtype=float)

                if uncertainty:  # find x,y,z error bars for relevant states

                    for count, state in enumerate(track):

                        if not count % err_freq:  # ie count % err_freq = 0
                            HH = np.eye(track.ndim)[mapping, :]  # Get position mapping matrix
                            cov = HH @ state.covar @ HH.T

                            err_x[count] = np.sqrt(cov[0, 0])
                            err_y[count] = np.sqrt(cov[1, 1])
                            err_z[count] = np.sqrt(cov[2, 2])

                self.fig.add_scatter3d(
                    x=[float(getattr(state, 'mean', state.state_vector)[mapping[0]])
                       for state in track],
                    error_x=dict(type='data', thickness=10, width=3, array=err_x),

                    y=[float(getattr(state, 'mean', state.state_vector)[mapping[1]])
                       for state in track],
                    error_y=dict(type='data', thickness=10, width=3, array=err_y),

                    z=[float(getattr(state, 'mean', state.state_vector)[mapping[2]])
                       for state in track],
                    error_z=dict(type='data', thickness=10, width=3, array=err_z),
                    # note that 3D error thickness seems to be broken in Plotly

                    text=[self._format_state_text(state) for state in track],
                    **scatter_kwargs)

            track_colors[track] = (self.fig.data[-1].line.color
                                   or self.fig.data[-1].marker.color
                                   or self.get_next_color())

        # earlier checking means this only applies to 2D.
        if uncertainty and self.dimension == 2:
            name = track_kwargs['legendgroup'] + "<br>(Ellipses)"
            add_legend = name not in {trace.legendgroup for trace in self.fig.data}
            for track in tracks:
                ellipse_kwargs = dict(
                    mode='none', fill='toself', fillcolor=track_colors[track],
                    opacity=0.2, hoverinfo='skip',
                    legendgroup=name, name=name,
                    legendrank=track_kwargs['legendrank'] + 10)
                for state in track:
                    points = self._generate_ellipse_points(state, mapping, ellipse_points)
                    if add_legend:
                        ellipse_kwargs['showlegend'] = True
                        add_legend = False
                    else:
                        ellipse_kwargs['showlegend'] = False

                    self.fig.add_scatter(x=points[0, :], y=points[1, :], **ellipse_kwargs)

        if particle and self.dimension == 2:
            name = track_kwargs['legendgroup'] + "<br>(Particles)"
            add_legend = name not in {trace.legendgroup for trace in self.fig.data}
            for track in tracks:
                for state in track:
                    particle_kwargs = dict(
                        mode='markers', marker=dict(size=2),
                        opacity=0.4, hoverinfo='skip',
                        legendgroup=name, name=name,
                        legendrank=track_kwargs['legendrank'] + 20)
                    if add_legend:
                        particle_kwargs['showlegend'] = True
                        add_legend = False
                    else:
                        particle_kwargs['showlegend'] = False
                    data = state.state_vector[mapping[:2], :]
                    self.fig.add_scattergl(x=data[0], y=data[1], **particle_kwargs)

    @staticmethod
    def _generate_ellipse_points(state, mapping, n_points=30):
        """Generate error ellipse points for given state and mapping"""
        HH = np.eye(state.ndim)[mapping, :]  # Get position mapping matrix
        w, v = np.linalg.eig(HH @ state.covar @ HH.T)
        max_ind = np.argmax(w)
        min_ind = np.argmin(w)
        orient = np.arctan2(v[1, max_ind], v[0, max_ind])
        a = np.sqrt(w[max_ind])
        b = np.sqrt(w[min_ind])
        m = 1 - (b**2 / a**2)

        def func(x):
            return np.sqrt(1 - (m**2 * np.sin(x)**2))

        def func2(z):
            return quad(func, 0, z)[0]

        c = 4 * a * func2(np.pi / 2)

        points = []
        for n in range(n_points):
            def func3(x):
                return n/n_points*c - a*func2(x)

            points.append((brentq(func3, 0, 2 * np.pi, xtol=1e-4)))

        c, s = np.cos(orient), np.sin(orient)
        rotational_matrix = np.array(((c, -s), (s, c)))
        points.append(points[0])
        points = np.array([[a * np.sin(i), b * np.cos(i)] for i in points])
        points = rotational_matrix @ points.T
        return points + state.mean[mapping[:2], :]

    def plot_sensors(self, sensors, mapping=[0, 1], sensor_label="Sensors", **kwargs):
        """Plots sensor(s)

        Plots sensors. Users can change the color and marker of sensors using keyword
        arguments. Default is a black 'x' marker.

        Parameters
        ----------
        sensors : Collection of :class:`~.Sensor`
            Sensors to plot
        mapping: list
            List of items specifying the mapping of the position
            components of the sensor's position.
        sensor_label: str
            Label to apply to all sensors for legend.
        \\*\\*kwargs: dict
            Additional arguments to be passed to scatter function for sensors. Defaults are
            ``marker=dict(symbol='x', color='black')``.
        """

        if not isinstance(sensors, Collection):
            sensors = {sensors}

        self._check_mapping(mapping)  # ensure mapping is compatible with plotter dimension

        if self.dimension == 1 or self.dimension == 3:
            raise NotImplementedError

        sensor_kwargs = dict(mode='markers', marker=dict(symbol='x', color='black'),
                             legendgroup=sensor_label, legendrank=50)
        sensor_kwargs.update(kwargs)

        sensor_kwargs['name'] = sensor_label
        if sensor_kwargs['legendgroup'] not in {trace.legendgroup
                                                for trace in self.fig.data}:
            sensor_kwargs['showlegend'] = True
        else:
            sensor_kwargs['showlegend'] = True

        sensor_xy = np.array([sensor.position[mapping, 0] for sensor in sensors])
        self.fig.add_scatter(x=sensor_xy[:, 0], y=sensor_xy[:, 1], **sensor_kwargs)

    def hide_plot_traces(self, items_to_hide: set):
        """Hide Plot Traces

        This function allows plotting items to be invisible as default. Users can toggle the plot
        trace to visible.

        Parameters
        ----------
        items_to_hide : set[str]
            The legend label (`legendgroups`) for the plot traces that should be invisible as
            default
        """
        for fig_data in self.fig.data:
            if fig_data.legendgroup in items_to_hide:
                fig_data.visible = "legendonly"
            else:
                fig_data.visible = None


class PolarPlotterly(_Plotter):

    def __init__(self, dimension=Dimension.TWO, **kwargs):
        if go is None:
            raise RuntimeError("Usage of Plotterly plotter requires installation of `plotly`")
        if isinstance(dimension, type(Dimension.TWO)):
            self.dimension = dimension
        elif isinstance(dimension, int):
            self.dimension = Dimension(dimension)
        else:
            raise TypeError("%s is an unsupported type for \'dimension\'; "
                            "expected type %s" % (type(dimension), type(Dimension.TWO)))
        if self.dimension != dimension.TWO:
            raise TypeError("Only 2D plotting currently supported")

        layout_kwargs = dict()
        layout_kwargs.update(kwargs)

        # Generate plot axes
        self.fig = go.Figure(layout=layout_kwargs)

    def plot_state_sequence(self, state_sequences, angle_mapping: int, range_mapping: int = None,
                            label="", **kwargs):
        """Plots state sequence(s)

        Plots each state sequence passed in to :attr:`state_sequences` and generates a legend
        automatically.

        Users can change line style, color and marker using keyword arguments. Any changes
        will apply to all ground truths.

        Parameters
        ----------
        state_sequences : Collection of :class:`~.StateMutableSequence`
            Collection of  state sequences which will be plotted. If not a collection,
            and instead a single :class:`~.StateMutableSequence` type, the argument is modified
            to be a set to allow for iteration.
        angle_mapping: int
            Specifying the mapping of the angular component of the state space to be plotted.
        range_mapping: int
            Specifying the mapping of the range component of the state space to be plotted. If
            `None`, the angular component will be plotted against time.
        label: str
            Label for truth data.
        \\*\\*kwargs: dict
            Additional arguments to be passed to scatter function. Default is
            ``mode=marker``.
            The default unit for the angular component is radians. This can be changed to degrees
            with the keyword argument ``thetaunit='degrees'``.
        """

        if not isinstance(state_sequences, Collection) \
                or isinstance(state_sequences, StateMutableSequence):
            state_sequences = {state_sequences}

        plotting_kwargs = dict(
            mode="markers", legendgroup=label, legendrank=200,
            name=label, thetaunit="radians")
        plotting_kwargs.update(kwargs)
        add_legend = plotting_kwargs['legendgroup'] not in {trace.legendgroup
                                                            for trace in self.fig.data}

        for state_sequence in state_sequences:
            if range_mapping is None:
                r = [state.timestamp for state in state_sequence]
            else:
                r = [float(state.state_vector[range_mapping]) for state in state_sequence]
            bearings = [float(state.state_vector[angle_mapping]) for state in state_sequence]

            scatter_kwargs = plotting_kwargs.copy()
            if add_legend:
                scatter_kwargs['showlegend'] = True
                add_legend = False
            else:
                scatter_kwargs['showlegend'] = False

            polar_plot = go.Scatterpolar(
                r=r,
                theta=bearings, **scatter_kwargs)
            self.fig.add_trace(polar_plot)

    def plot_ground_truths(self, truths, mapping, truths_label="Ground Truth", **kwargs):
        """Plots ground truth(s)

        Plots each ground truth path passed in to :attr:`truths` and generates a legend
        automatically. Ground truths are plotted as dashed lines with default colors.

        Users can change line style, color and marker using keyword arguments. Any changes
        will apply to all ground truths.

        Parameters
        ----------
        truths : Collection of :class:`~.GroundTruthPath`
            Collection of  ground truths which will be plotted. If not a collection,
            and instead a single :class:`~.GroundTruthPath` type, the argument is modified to be a
            set to allow for iteration.
        mapping: list
            List of items specifying the mapping of the position components of the state space.
        truths_label: str
            Label for truth data. Default is "Ground Truth".
        \\*\\*kwargs: dict
            Additional arguments to be passed to scatter function. Default is
            ``line=dict(dash="dash")``.
        """
        truths_kwargs = dict(mode="lines", line=dict(dash="dash"), legendrank=100)
        truths_kwargs.update(kwargs)
        angle_mapping = mapping[0]
        if len(mapping) > 1:
            range_mapping = mapping[1]
        else:
            range_mapping = None
        self.plot_state_sequence(state_sequences=truths, angle_mapping=angle_mapping,
                                 range_mapping=range_mapping, label=truths_label, **truths_kwargs)

    def plot_measurements(self, measurements, mapping, measurement_model=None,
                          measurements_label="Measurements", convert_measurements=True, **kwargs):
        """Plots measurements

        Plots detections and clutter, generating a legend automatically. Detections are plotted as
        blue circles by default unless the detection type is clutter.
        If the detection type is :class:`~.Clutter` it is plotted as a yellow 'tri-up' marker.

        Users can change the color and marker of detections using keyword arguments but not for
        clutter detections.

        Parameters
        ----------
        measurements : Collection of :class:`~.Detection`
            Detections which will be plotted. If measurements is a set of lists it is flattened.
        mapping: list
            List of items specifying the mapping of the position components of the state space.
        measurement_model : :class:`~.Model`, optional
            User-defined measurement model to be used in finding measurement state inverses if
            they cannot be found from the measurements themselves.
        measurements_label : str
            Label for the measurements.  Default is "Measurements".
        convert_measurements: bool
            Should the measurements be converted before being plotted. Default is True.
        \\*\\*kwargs: dict
            Additional arguments to be passed to scatter function for detections. Defaults are
            ``marker=dict(color="#636EFA")``.
        """

        if not isinstance(measurements, Collection):
            measurements = {measurements}

        if any(isinstance(item, set) for item in measurements):
            measurements_set = chain.from_iterable(measurements)  # Flatten into one set
        else:
            measurements_set = set(measurements)

        plot_detections, plot_clutter = self._conv_measurements(measurements_set,
                                                                mapping,
                                                                measurement_model,
                                                                convert_measurements)

        angle_mapping = 0
        if len(mapping) > 1:
            range_mapping = 1
        else:
            range_mapping = None

        if plot_detections:
            name = measurements_label + "<br>(Detections)"
            measurement_kwargs = dict(mode='markers', marker=dict(color='#636EFA'), legendrank=200)
            measurement_kwargs.update(kwargs)
            plotting_data = [State(state_vector=plotting_state_vector,
                                   timestamp=det.timestamp)
                             for det, plotting_state_vector in plot_detections.items()]

            self.plot_state_sequence(state_sequences=[plotting_data], angle_mapping=angle_mapping,
                                     range_mapping=range_mapping, label=name,
                                     **measurement_kwargs)

        if plot_clutter:
            name = measurements_label + "<br>(Clutter)"
            measurement_kwargs = dict(mode='markers', legendrank=210,
                                      marker=dict(symbol="star-triangle-up", color='#FECB52'))
            measurement_kwargs.update(kwargs)
            plotting_data = [State(state_vector=plotting_state_vector,
                                   timestamp=det.timestamp)
                             for det, plotting_state_vector in plot_clutter.items()]

            self.plot_state_sequence(state_sequences=[plotting_data], angle_mapping=angle_mapping,
                                     range_mapping=range_mapping, label=name,
                                     **measurement_kwargs)

    def plot_tracks(self, tracks, mapping, uncertainty=False, particle=False, track_label="Tracks",
                    **kwargs):
        """Plots track(s)

        Plots each track generated, generating a legend automatically. If ``uncertainty=True``
        error ellipses are plotted.
        Tracks are plotted as solid lines with point markers and default colors.

        Users can change line style, color and marker using keyword arguments.

        Parameters
        ----------
        tracks : Collection of :class:`~.Track`
            Collection of tracks which will be plotted. If not a collection, and instead a single
            :class:`~.Track` type, the argument is modified to be a set to allow for iteration.
        mapping: list
            List of items specifying the mapping of the position
            components of the state space.
        uncertainty : bool
            If True, function plots uncertainty ellipses.
        particle : bool
            If True, function plots particles.
        track_label: str
            Label to apply to all tracks for legend.
        \\*\\*kwargs: dict
            Additional arguments to be passed to scatter function. Defaults are
            ``mode='markers+lines'``.
        """
        if uncertainty or particle:
            raise NotImplementedError

        track_kwargs = dict(mode='markers+lines', legendrank=300)
        track_kwargs.update(kwargs)
        angle_mapping = mapping[0]
        if len(mapping) > 1:
            range_mapping = mapping[1]
        else:
            range_mapping = None
        self.plot_state_sequence(state_sequences=tracks, angle_mapping=angle_mapping,
                                 range_mapping=range_mapping, label=track_label, **track_kwargs)

    def plot_sensors(self, sensors, sensor_label="Sensors", **kwargs):
        raise NotImplementedError


class _AnimationPlotterDataClass(Base):
    plotting_data = Property(Iterable[State])
    plotting_label: str = Property()
    plotting_keyword_arguments: dict = Property()


class AnimationPlotter(_Plotter):

    def __init__(self, dimension=Dimension.TWO, x_label: str = "$x$", y_label: str = "$y$",
                 title: str = None, legend_kwargs: dict = {}, **kwargs):

        self.figure_kwargs = {"figsize": (10, 6)}
        self.figure_kwargs.update(kwargs)
        if dimension != Dimension.TWO:
            raise NotImplementedError

        self.legend_kwargs = dict()
        self.legend_kwargs.update(legend_kwargs)

        self.x_label: str = x_label
        self.y_label: str = y_label

        if title:
            title += "\n"
        self.title: str = title

        self.plotting_data: List[_AnimationPlotterDataClass] = []

        self.animation_output: animation.FuncAnimation = None

    def run(self,
            times_to_plot: List[datetime] = None,
            plot_item_expiry: Optional[timedelta] = None,
            **kwargs):
        """Run the animation

        Parameters
        ----------
        times_to_plot : List of :class:`~.datetime`
            List of datetime objects of when to refresh and draw the animation. Default `None`,
            where unique timestamps of data will be used.
        plot_item_expiry: :class:`~.timedelta`, Optional
            Describes how long states will remain present in the figure. Default value of None
            means data is shown indefinitely
        \\*\\*kwargs: dict
            Additional arguments to be passed to the animation.FuncAnimation function
        """
        if times_to_plot is None:
            times_to_plot = sorted({
                state.timestamp
                for plotting_data in self.plotting_data
                for state in plotting_data.plotting_data})

        self.animation_output = self.run_animation(
            times_to_plot=times_to_plot,
            data=self.plotting_data,
            plot_item_expiry=plot_item_expiry,
            x_label=self.x_label,
            y_label=self.y_label,
            figure_kwargs=self.figure_kwargs,
            legend_kwargs=self.legend_kwargs,
            animation_input_kwargs=kwargs,
            plot_title=self.title
        )
        return self.animation_output

    def save(self, filename='example.mp4', **kwargs):
        """Save the animation

        Parameters
        ----------
        filename : str
            filename of animation file
        \\*\\*kwargs: dict
            Additional arguments to be passed to the animation.save function
        """
        if self.animation_output is None:
            raise ValueError("Animation hasn't been run yet. Therefore there is no animation to "
                             "save")

        self.animation_output.save(filename, **kwargs)

    def plot_ground_truths(self, truths, mapping: List[int], truths_label: str = "Ground Truth",
                           **kwargs):
        """Plots ground truth(s)

        Plots each ground truth path passed in to :attr:`truths` and generates a legend
        automatically. Ground truths are plotted as dashed lines with default colors.

        Users can change linestyle, color and marker using keyword arguments. Any changes
        will apply to all ground truths.

        Parameters
        ----------
        truths : Collection of :class:`~.GroundTruthPath`
            Collection of  ground truths which will be plotted. If not a collection and instead a
            single :class:`~.GroundTruthPath` type, the argument is modified to be a set to allow
            for iteration.
        mapping: list
            List of items specifying the mapping of the position components of the state space.
        truths_label: str
            Label for truth data. Default is "Ground Truth"
        \\*\\*kwargs: dict
            Additional arguments to be passed to plot function. Default is ``linestyle="--"``.
        """

        truths_kwargs = dict(linestyle="--")
        truths_kwargs.update(kwargs)
        self.plot_state_mutable_sequence(truths, mapping, truths_label, **truths_kwargs)

    def plot_tracks(self, tracks, mapping: List[int], uncertainty=False, particle=False,
                    track_label="Tracks", **kwargs):
        """Plots track(s)

        Plots each track generated, generating a legend automatically. Tracks are plotted as solid
        lines with point markers and default colors. Users can change linestyle, color and marker
        using keyword arguments.

        Parameters
        ----------
        tracks : Collection of :class:`~.Track`
            Collection of tracks which will be plotted. If not a collection, and instead a single
            :class:`~.Track` type, the argument is modified to be a set to allow for iteration.
        mapping: list
            List of items specifying the mapping of the position
            components of the state space.
        uncertainty : bool
            Currently not implemented. If True, an error is raised
        particle : bool
            Currently not implemented. If True, an error is raised
        track_label: str
            Label to apply to all tracks for legend.
        \\*\\*kwargs: dict
            Additional arguments to be passed to plot function. Defaults are ``linestyle="-"``,
            ``marker='s'`` for :class:`~.Update` and ``marker='o'`` for other states.
        """
        if uncertainty or particle:
            raise NotImplementedError

        tracks_kwargs = dict(linestyle='-', marker="s", color=None)
        tracks_kwargs.update(kwargs)
        self.plot_state_mutable_sequence(tracks, mapping, track_label, **tracks_kwargs)

    def plot_state_mutable_sequence(self, state_mutable_sequences, mapping: List[int], label: str,
                                    **plotting_kwargs):
        """Plots State Mutable Sequence

        Parameters
        ----------
        state_mutable_sequences : Collection of :class:`~.StateMutableSequence`
            Collection of states to be plotted
        mapping: list
            List of items specifying the mapping of the position components of the state space.
        label : str
            User-defined measurement model to be used in finding measurement state inverses if
            they cannot be found from the measurements themselves.
        \\*\\*kwargs: dict
            Additional arguments to be passed to plot function for states.
        """

        if not isinstance(state_mutable_sequences, Collection) or \
                isinstance(state_mutable_sequences, StateMutableSequence):
            state_mutable_sequences = {state_mutable_sequences}  # Make a set of length 1

        for idx, state_mutable_sequence in enumerate(state_mutable_sequences):
            if idx == 0:
                this_plotting_label = label
            else:
                this_plotting_label = None

            self.plotting_data.append(_AnimationPlotterDataClass(
                plotting_data=[State(state_vector=[state.state_vector[mapping[0]],
                                                   state.state_vector[mapping[1]]],
                                     timestamp=state.timestamp)
                               for state in state_mutable_sequence],
                plotting_label=this_plotting_label,
                plotting_keyword_arguments=plotting_kwargs
            ))

    def plot_measurements(self, measurements, mapping, measurement_model=None,
                          measurements_label="", convert_measurements=True, **kwargs):
        """Plots measurements

        Plots detections and clutter, generating a legend automatically. Detections are plotted as
        blue circles by default unless the detection type is clutter.
        If the detection type is :class:`~.Clutter` it is plotted as a yellow 'tri-up' marker.

        Users can change the color and marker of detections using keyword arguments but not for
        clutter detections.

        Parameters
        ----------
        measurements : Collection of :class:`~.Detection`
            Detections which will be plotted. If measurements is a set of lists it is flattened.
        mapping: list
            List of items specifying the mapping of the position components of the state space.
        measurement_model : :class:`~.Model`, optional
            User-defined measurement model to be used in finding measurement state inverses if
            they cannot be found from the measurements themselves.
        measurements_label: str
            Label for measurements. Default will be "Detections" or "Clutter"
        convert_measurements: bool
            Should the measurements be converted from measurement space to state space before
            being plotted. Default is True
        \\*\\*kwargs: dict
            Additional arguments to be passed to plot function for detections. Defaults are
            ``marker='o'`` and ``color='b'``.
        """

        measurement_kwargs = dict(marker='o', color='b')
        measurement_kwargs.update(kwargs)

        if not isinstance(measurements, Collection):
            measurements = {measurements}  # Make a set of length 1

        if any(isinstance(item, set) for item in measurements):
            measurements_set = chain.from_iterable(measurements)  # Flatten into one set
        else:
            measurements_set = measurements

        plot_detections, plot_clutter = self._conv_measurements(measurements_set,
                                                                mapping,
                                                                measurement_model,
                                                                convert_measurements)

        if measurements_label != "":
            measurements_label = measurements_label + " "

        if plot_detections:
            detection_kwargs = dict(linestyle='', marker='o', color='b')
            detection_kwargs.update(kwargs)
            self.plotting_data.append(_AnimationPlotterDataClass(
                plotting_data=[State(state_vector=plotting_state_vector,
                                     timestamp=detection.timestamp)
                               for detection, plotting_state_vector in plot_detections.items()],
                plotting_label=measurements_label + "Detections",
                plotting_keyword_arguments=detection_kwargs
            ))

        if plot_clutter:
            clutter_kwargs = dict(linestyle='', marker='2', color='y')
            clutter_kwargs.update(kwargs)
            self.plotting_data.append(_AnimationPlotterDataClass(
                plotting_data=[State(state_vector=plotting_state_vector,
                                     timestamp=detection.timestamp)
                               for detection, plotting_state_vector in plot_clutter.items()],
                plotting_label=measurements_label + "Clutter",
                plotting_keyword_arguments=clutter_kwargs
            ))

    def plot_sensors(self, sensors, sensor_label="Sensors", **kwargs):
        raise NotImplementedError

    @classmethod
    def run_animation(cls,
                      times_to_plot: List[datetime],
                      data: Iterable[_AnimationPlotterDataClass],
                      plot_item_expiry: Optional[timedelta] = None,
                      axis_padding: float = 0.1,
                      figure_kwargs: dict = {},
                      animation_input_kwargs: dict = {},
                      legend_kwargs: dict = {},
                      x_label: str = "$x$",
                      y_label: str = "$y$",
                      plot_title: str = None
                      ) -> animation.FuncAnimation:
        """
        Parameters
        ----------
        times_to_plot : Iterable[datetime]
            All the times that the plotter should plot
        data : Iterable[datetime]
            All the data that should be plotted
        plot_item_expiry: timedelta
            How long a state should be displayed for. Default value of None
            means data is shown indefinitely
        axis_padding: float
            How much extra space should be given around the edge of the plot
        figure_kwargs: dict
            Keyword arguments for the pyplot figure function. See matplotlib.pyplot.figure for more
            details
        animation_input_kwargs: dict
            Keyword arguments for FuncAnimation class. See matplotlib.animation.FuncAnimation for
            more details. Default values are: blit=False, repeat=False, interval=50
        legend_kwargs: dict
            Keyword arguments for the pyplot legend function. See matplotlib.pyplot.legend for more
            details
        x_label: str
            Label for the x axis
        y_label: str
            Label for the y axis
        plot_title: str
            Title for the plot

        Returns
        -------
        : animation.FuncAnimation
            Animation object
        """

        animation_kwargs = dict(blit=False, repeat=False, interval=50)  # milliseconds
        animation_kwargs.update(animation_input_kwargs)

        fig1 = plt.figure(**figure_kwargs)

        the_lines = []
        plotting_data = []
        legends_key = []

        for a_plot_object in data:
            if a_plot_object.plotting_data is not None:
                the_data = np.array(
                    [a_state.state_vector for a_state in a_plot_object.plotting_data])
                if len(the_data) == 0:
                    continue
                the_lines.append(
                    plt.plot([],  # the_data[:1, 0],
                             [],  # the_data[:1, 1],
                             **a_plot_object.plotting_keyword_arguments)[0])

                legends_key.append(a_plot_object.plotting_label)
                plotting_data.append(a_plot_object.plotting_data)

        if axis_padding:
            [x_limits, y_limits] = [
                [min(state.state_vector[idx] for line in data for state in line.plotting_data),
                 max(state.state_vector[idx] for line in data for state in line.plotting_data)]
                for idx in [0, 1]]

            for axis_limits in [x_limits, y_limits]:
                limit_padding = axis_padding * (axis_limits[1] - axis_limits[0])
                # The casting to float to ensure the limits contain do not contain angle classes
                axis_limits[0] = float(axis_limits[0] - limit_padding)
                axis_limits[1] = float(axis_limits[1] + limit_padding)

            plt.xlim(x_limits)
            plt.ylim(y_limits)
        else:
            plt.axis('equal')

        plt.xlabel(x_label)
        plt.ylabel(y_label)

        lines_with_legend = [line for line, label in zip(the_lines, legends_key)
                             if label is not None]
        plt.legend(lines_with_legend, [label for label in legends_key if label is not None],
                   **legend_kwargs)

        if plot_item_expiry is None:
            min_plot_time = min(state.timestamp
                                for line in data
                                for state in line.plotting_data)
            min_plot_times = [min_plot_time] * len(times_to_plot)
        else:
            min_plot_times = [time - plot_item_expiry for time in times_to_plot]

        line_ani = animation.FuncAnimation(fig1, cls.update_animation,
                                           frames=len(times_to_plot),
                                           fargs=(the_lines, plotting_data, min_plot_times,
                                                  times_to_plot, plot_title),
                                           **animation_kwargs)

        plt.draw()

        return line_ani

    @staticmethod
    def update_animation(index: int, lines: List[Line2D], data_list: List[List[State]],
                         start_times: List[datetime], end_times: List[datetime], title: str):
        """
        Parameters
        ----------
        index : int
            Which index of the start_times and end_times should be used
        lines : List[Line2D]
            The data that will be plotted, to be plotted.
        data_list : List[List[State]]
            All the data that should be plotted
        start_times : List[datetime]
            lowest (earliest) time for an item to be plotted
        end_times : List[datetime]
            highest (latest) time for an item to be plotted
        title: str
            Title for the plot

        Returns
        -------
        : List[Line2D]
            The data that will be plotted
        """

        min_time = start_times[index]
        max_time = end_times[index]

        if title is None:
            title = ""
        plt.title(title + str(max_time))
        for i, data_source in enumerate(data_list):

            if data_source is not None:
                the_data = np.array([a_state.state_vector for a_state in data_source
                                     if min_time <= a_state.timestamp <= max_time])
                if the_data.size > 0:
                    lines[i].set_data(the_data[:, 0],
                                      the_data[:, 1])
                else:
                    lines[i].set_data([],
                                      [])
        return lines


class AnimatedPlotterly(_Plotter):
    """
    Class for a 2D animated plotter that uses Plotly graph objects rather than matplotlib.
    This gives the user the ability to see how tracking works through time, while being
    able to interact with tracks, truths, etc, in the same way that is enabled by
    Plotly static plots.

    Simplifies the process of plotting ground truths, measurements, clutter, and tracks.
    Tracks can be plotted with uncertainty ellipses or particles if required. Legends
    are automatically generated with each plot.

    Parameters
    ----------
    timesteps: Collection
        Collection of equally-spaced timesteps. Each animation frame is a timestep.
    tail_length: float
        Percentage of sim time for which previous values will still be displayed for.
        Value can be between 0 and 1. Default is 0.3.
    equal_size: bool
        Makes x and y axes equal when figure is resized. Default is False.
    sim_duration: int
        Time taken to run animation (s). Default is 6
    \\*\\*kwargs
        Additional arguments to be passed in the initialisation.

    Attributes
    ----------

    """

    def __init__(self, timesteps, tail_length=0.3, equal_size=False,
                 sim_duration=6, **kwargs):
        """
        Initialise the figure and checks that inputs are correctly formatted.
        Creates an empty frame for each timestep, and configures
        the buttons and slider.


        """
        if go is None or colors is None:
            raise RuntimeError("Usage of Plotterly plotter requires installation of `plotly`")

        self.equal_size = equal_size

        # checking that there are multiple timesteps
        if len(timesteps) < 2:
            raise ValueError("Must be at least 2 timesteps for animation.")

        # checking that timesteps are evenly spaced
        time_spaces = np.unique(np.diff(timesteps))

        # gives the unique values of time gaps between timesteps. If this contains more than
        # one value, then timesteps are not all evenly spaced which is an issue.
        if len(time_spaces) != 1:
            warnings.warn("Timesteps are not equally spaced, so the passage of time is not linear")
        self.timesteps = timesteps

        # checking input to tail_length
        if tail_length > 1 or tail_length < 0:
            raise ValueError("Tail length should be between 0 and 1")
        self.tail_length = tail_length

        # checking sim_duration
        if sim_duration <= 0:
            raise ValueError("Simulation duration must be positive")

        # time window is calculated as sim_length * tail_length. This is
        # the window of time for which past plots are still visible
        self.time_window = (timesteps[-1] - timesteps[0]) * tail_length

        self.colorway = colors.qualitative.Plotly[1:]  # plotting colours

        self.all_masks = dict()  # dictionary to be filled up later

        self.plotting_function_called = False  # keeps track if anything has been plotted or not
        # so that only the first data plotted will override the default axis max and mins.

        self.fig = go.Figure()

        layout_kwargs = dict(
            xaxis=dict(title=dict(text="<i>x</i>")),
            yaxis=dict(title=dict(text="<i>y</i>")),
            colorway=self.colorway,  # Needed to match colours later.
            height=550,
            autosize=True
        )
        # layout_kwargs.update(kwargs)
        self.fig.update_layout(layout_kwargs)

        # initialise frames according to simulation timesteps
        self.fig.frames = [dict(
            name=str(time),
            data=[],
            traces=[]
        ) for time in timesteps]

        self.fig.update_xaxes(range=[0, 10])
        self.fig.update_yaxes(range=[0, 10])

        frame_duration = sim_duration * 1000 / len(self.fig.frames)

        # if the gap between timesteps is greater than a day, it isn't necessary
        # to display hour and minute information, so remove this to give a cleaner display.
        # a and b are used in the slider steps label later
        if time_spaces[0] >= timedelta(days=1):
            start_cut_off = None
            end_cut_off = 10

        # if the simulation is over a day long, display all information which
        # looks clunky but is necessary
        elif timesteps[-1] - timesteps[0] > timedelta(days=1):
            start_cut_off = None
            end_cut_off = None

        # otherwise, remove day information and just show
        # hours, mins, etc. which is cleaner to look at
        else:
            start_cut_off = 11
            end_cut_off = None

        # create button and slider
        updatemenus = [dict(type='buttons',
                            buttons=[{
                                "args": [None,
                                         {"frame": {"duration": frame_duration, "redraw": True},
                                          "fromcurrent": True, "transition": {"duration": 0}}],
                                "label": "Play",
                                "method": "animate"
                            }, {
                                "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                                  "mode": "immediate",
                                                  "transition": {"duration": 0}}],
                                "label": "Stop",
                                "method": "animate"
                            }],
                            direction='left',
                            pad=dict(r=10, t=75),
                            showactive=True, x=0.1, y=0, xanchor='right', yanchor='top')
                       ]
        sliders = [{'yanchor': 'top',
                    'xanchor': 'left',
                    'currentvalue': {'font': {'size': 16}, 'prefix': 'Time: ', 'visible': True,
                                     'xanchor': 'right'},
                    'transition': {'duration': frame_duration, 'easing': 'linear'},
                    'pad': {'b': 10, 't': 50},
                    'len': 0.9, 'x': 0.1, 'y': 0,
                    'steps': [{'args': [[frame.name], {
                        'frame': {'duration': 1.0, 'easing': 'linear', 'redraw': True},
                        'transition': {'duration': 0, 'easing': 'linear'}}],
                               'label': frame.name[start_cut_off: end_cut_off],
                               'method': 'animate'} for frame in
                              self.fig.frames
                              ]}]
        self.fig.update_layout(updatemenus=updatemenus, sliders=sliders)
        self.fig.update_layout(kwargs)

    def show(self):
        """
        Display the animation.
        """
        return self.fig

    def _resize(self, data, type="track"):
        """
        Reshape figure so that everything is in view.

        Parameters
        ----------

        data:
            Collection of values that are being added to the figure.
            Will be a list if coming from plot_ground_Truths or
            plot_tracks, but will be a dictionary if coming from plot_measurements.
        """

        # fill in all data. If there is no data, fill all_x, all_y with current axis limits
        if not data:
            all_x = list(self.fig.layout.xaxis.range)
            all_y = list(self.fig.layout.xaxis.range)
        else:
            all_x = list()
            all_y = list()

        # fill in data
        if type == "measurements":

            for key, item in data.items():
                all_x.extend(data[key]["x"])
                all_y.extend(data[key]["y"])

        elif type in ("ground_truth", "tracks"):

            for n, _ in enumerate(data):
                all_x.extend(data[n]["x"])
                all_y.extend(data[n]["y"])

        elif type == "sensor":
            sensor_xy = np.array([sensor.position[[0, 1], 0] for sensor in data])
            all_x.extend(sensor_xy[:, 0])
            all_y.extend(sensor_xy[:, 1])

        elif type == "particle_or_uncertainty":
            # data comes in format of list of dictionaries. Each dictionary contains 'x' and 'y',
            # which are a list of lists.
            for dictionary in data:
                for x_values in dictionary["x"]:
                    all_x.extend([np.nanmax(x_values), np.nanmin(x_values)])
                for y_values in dictionary["y"]:
                    all_y.extend([np.nanmax(y_values), np.nanmin(y_values)])

        xmax = max(all_x)
        ymax = max(all_y)
        xmin = min(all_x)
        ymin = min(all_y)

        if self.equal_size:
            xmax = ymax = max(xmax, ymax)
            xmin = ymin = min(xmin, ymin)

        # if it's first time plotting data, want to ensure plotter is bound to that data
        # and not the default values. Issues arise if the initial plotted data is much
        # smaller than the default 0 to 10 values.
        if not self.plotting_function_called:

            self.fig.update_xaxes(range=[xmin, xmax])
            self.fig.update_yaxes(range=[ymin, ymax])

        # need to check if it's actually necessary to resize or not
        if xmax >= self.fig.layout.xaxis.range[1] or xmin <= self.fig.layout.xaxis.range[0]:

            xmax = max(xmax, self.fig.layout.xaxis.range[1])
            xmin = min(xmin, self.fig.layout.xaxis.range[0])
            xrange = xmax - xmin

            # update figure while adding a small buffer to the mins and maxes
            self.fig.update_xaxes(range=[xmin - xrange / 20, xmax + xrange / 20])

        if ymax >= self.fig.layout.yaxis.range[1] or ymin <= self.fig.layout.yaxis.range[0]:

            ymax = max(ymax, self.fig.layout.yaxis.range[1])
            ymin = min(ymin, self.fig.layout.yaxis.range[0])
            yrange = ymax - ymin

            self.fig.update_yaxes(range=[ymin - yrange / 20, ymax + yrange / 20])

    def plot_ground_truths(self, truths, mapping, truths_label="Ground Truth",
                           resize=True, **kwargs):

        """Plots ground truth(s)

        Plots each ground truth path passed in to :attr:`truths` and generates a legend
        automatically. Ground truths are plotted as dashed lines with default colors.

        Users can change linestyle, color and marker using keyword arguments. Any changes
        will apply to all ground truths.

        Parameters
        ----------
        truths : Collection of :class:`~.GroundTruthPath`
            Collection of  ground truths which will be plotted. If not a collection and instead a
            single :class:`~.GroundTruthPath` type, the argument is modified to be a set to allow
            for iteration.
        mapping: list
            List of items specifying the mapping of the position components of the state space.
        truths_label: str
            Name of ground truths in legend/plot
        resize: bool
            if True, will resize figure to ensure that ground truths are in view
        \\*\\*kwargs: dict
            Additional arguments to be passed to plot function. Default is ``linestyle="--"``.

        """

        if not isinstance(truths, Collection) or isinstance(truths, StateMutableSequence):
            truths = {truths}  # Make a set of length 1

        data = [dict() for _ in truths]  # put all data into one place for later plotting
        for n, truth in enumerate(truths):

            # initialise arrays that go inside the dictionary
            data[n].update(x=np.zeros(len(truth)),
                           y=np.zeros(len(truth)),
                           time=np.array([0 for _ in range(len(truth))], dtype=object),
                           time_str=np.array([0 for _ in range(len(truth))], dtype=object),
                           type=np.array([0 for _ in range(len(truth))], dtype=object))

            for k, state in enumerate(truth):
                # fill the arrays here
                data[n]["x"][k] = state.state_vector[mapping[0]]
                data[n]["y"][k] = state.state_vector[mapping[1]]
                data[n]["time"][k] = state.timestamp
                data[n]["time_str"][k] = str(state.timestamp)
                data[n]["type"][k] = type(state).__name__

        trace_base = len(self.fig.data)  # number of traces currently in the animation

        # add a trace that keeps the legend up for the entire simulation (will remain
        # even if no truths are present), then add a trace for each truth in the simulation.
        # initialise keyword arguments, then add them to the traces
        truth_kwargs = dict(x=[], y=[], mode="lines", hoverinfo='none', legendgroup=truths_label,
                            line=dict(dash="dash", color=self.colorway[0]), legendrank=100,
                            name=truths_label, showlegend=True)
        truth_kwargs.update(kwargs)
        # legend dummy trace
        self.fig.add_trace(go.Scatter(truth_kwargs))

        # we don't want the legend for any of the actual traces
        truth_kwargs.update({"showlegend": False})

        for n, _ in enumerate(truths):
            # change the colour of each truth and include n in its name
            truth_kwargs.update({
                "line": dict(dash="dash", color=self.colorway[n % len(self.colorway)])})
            truth_kwargs.update(kwargs)
            self.fig.add_trace(go.Scatter(truth_kwargs))  # add to traces

        for frame in self.fig.frames:

            # get current fig data and traces
            data_ = list(frame.data)
            traces_ = list(frame.traces)

            # convert string to datetime object
            frame_time = datetime.fromisoformat(frame.name)
            cutoff_time = (frame_time - self.time_window)

            # for the legend
            data_.append(go.Scatter(x=[0, 0], y=[0, 0]))
            traces_.append(trace_base)

            for n, truth in enumerate(truths):
                # all truth points that come at or before the frame time
                t_upper = [data[n]["time"] <= frame_time]

                # only select detections that come after the time cut-off
                t_lower = [data[n]["time"] >= cutoff_time]

                # put together
                mask = np.logical_and(t_upper, t_lower)

                # find x, y, time, and type
                truth_x = data[n]["x"][tuple(mask)]
                # add in np.inf to ensure traces are present for every timestep
                truth_x = np.append(truth_x, [np.inf])
                truth_y = data[n]["y"][tuple(mask)]
                truth_y = np.append(truth_y, [np.inf])
                times = data[n]["time_str"][tuple(mask)]

                data_.append(go.Scatter(x=truth_x,
                                        y=truth_y,
                                        meta=times,
                                        hovertemplate='GroundTruthState' +
                                                      '<br>(%{x}, %{y})' +
                                                      '<br>Time: %{meta}'))

                traces_.append(trace_base + n + 1)  # append data to correct trace

                frame.data = data_
                frame.traces = traces_

        if resize:
            self._resize(data, type="ground_truth")

        # we have called a plotting function so update flag (gets used in _resize)
        self.plotting_function_called = True

    def plot_measurements(self, measurements, mapping, measurement_model=None,
                          resize=True, measurements_label="Measurements",
                          convert_measurements=True, **kwargs):
        """Plots measurements

        Plots detections and clutter, generating a legend automatically. Detections are plotted as
        blue circles by default unless the detection type is clutter.
        If the detection type is :class:`~.Clutter` it is plotted as a yellow 'tri-up' marker.

        Users can change the color and marker of detections using keyword arguments but not for
        clutter detections.

        Parameters
        ----------
        measurements : Collection of :class:`~.Detection`
            Detections which will be plotted. If measurements is a set of lists it is flattened.
        mapping: list
            List of items specifying the mapping of the position components of the state space.
        measurement_model : :class:`~.Model`, optional
            User-defined measurement model to be used in finding measurement state inverses if
            they cannot be found from the measurements themselves.
        resize: bool
            If True, will resize figure to ensure measurements are in view
        measurements_label : str
            Label for the measurements.  Default is "Measurements".
        convert_measurements : bool
            Should the measurements be converted from measurement space to state space before
            being plotted. Default is True
        \\*\\*kwargs: dict
            Additional arguments to be passed to scatter function for detections. Defaults are
            ``marker=dict(color="#636EFA")``.
        """

        if not isinstance(measurements, Collection):
            measurements = {measurements}  # Make a set of length 1

        if any(isinstance(item, set) for item in measurements):
            measurements_set = chain.from_iterable(measurements)  # Flatten into one set
        else:
            measurements_set = measurements
        plot_detections, plot_clutter = self._conv_measurements(measurements_set,
                                                                mapping,
                                                                measurement_model,
                                                                convert_measurements)
        plot_combined = {'Detection': plot_detections,
                         'Clutter': plot_clutter}  # for later reference

        # this dictionary will store all the plotting data that we need
        # from the detections and clutter into numpy arrays that we can easily
        # access to plot
        combined_data = dict()

        # only add clutter or detections to plot if necessary
        if plot_detections:
            combined_data.update(dict(Detection=dict()))
        if plot_clutter:
            combined_data.update(dict(Clutter=dict()))

        # initialise combined_data
        for key in combined_data.keys():
            length = len(plot_combined[key])
            combined_data[key].update({
                "x": np.zeros(length),
                "y": np.zeros(length),
                "time": np.array([0 for _ in range(length)], dtype=object),
                "time_str": np.array([0 for _ in range(length)], dtype=object),
                "type": np.array([0 for _ in range(length)], dtype=object)})

        # and now fill in the data

        for key in combined_data.keys():
            for n, det in enumerate(plot_combined[key]):
                x, y = list(plot_combined[key].values())[n]
                combined_data[key]["x"][n] = x
                combined_data[key]["y"][n] = y
                combined_data[key]["time"][n] = det.timestamp
                combined_data[key]["time_str"][n] = str(det.timestamp)
                combined_data[key]["type"][n] = type(det).__name__

        # get number of traces currently in fig
        trace_base = len(self.fig.data)

        # initialise detections
        name = measurements_label + "<br>(Detections)"
        measurement_kwargs = dict(x=[], y=[], mode='markers',
                                  name=name,
                                  legendgroup='Detections (Measurements)',
                                  legendrank=200, showlegend=True,
                                  marker=dict(color="#636EFA"), hoverinfo='none')
        measurement_kwargs.update(kwargs)

        self.fig.add_trace(go.Scatter(measurement_kwargs))  # trace for legend

        measurement_kwargs.update({"showlegend": False})
        self.fig.add_trace(go.Scatter(measurement_kwargs))  # trace for plotting

        # change necessary kwargs to initialise clutter trace
        name = measurements_label + "<br>(Clutter)"
        measurement_kwargs.update({"legendgroup": 'Clutter', "legendrank": 300,
                                   "marker": dict(symbol="star-triangle-up", color='#FECB52'),
                                   "name": name, 'showlegend': True})

        self.fig.add_trace(go.Scatter(measurement_kwargs))  # trace for plotting clutter

        # add data to frames
        for frame in self.fig.frames:

            data_ = list(frame.data)
            traces_ = list(frame.traces)

            # add blank data to ensure detection legend stays in place
            data_.append(go.Scatter(x=[-np.inf, np.inf], y=[-np.inf, np.inf]))
            traces_.append(trace_base)  # ensure data is added to correct trace

            frame_time = datetime.fromisoformat(frame.name)  # convert string to datetime object

            # time at which dets will disappear from the fig
            cutoff_time = (frame_time - self.time_window)

            for j, key in enumerate(combined_data.keys()):
                # only select measurements that arrive by the time of the current frame
                t_upper = [combined_data[key]["time"] <= frame_time]

                # only select detections that come after the time cut-off
                t_lower = [combined_data[key]["time"] >= cutoff_time]

                # put them together to create the final mask
                mask = np.logical_and(t_upper, t_lower)

                # find x and y points for true detections and clutter
                det_x = combined_data[key]["x"][tuple(mask)]
                det_x = np.append(det_x, [np.inf])
                det_y = combined_data[key]["y"][tuple(mask)]
                det_y = np.append(det_y, [np.inf])
                det_times = combined_data[key]["time_str"][tuple(mask)]

                data_.append(go.Scatter(x=det_x,
                                        y=det_y,
                                        meta=det_times,
                                        hovertemplate=f'{key}' +
                                                      '<br>(%{x}, %{y})' +
                                                      '<br>Time: %{meta}'))
                traces_.append(trace_base + j + 1)

            frame.data = data_  # update the figure
            frame.traces = traces_

        if resize:
            self._resize(combined_data, "measurements")

        # we have called a plotting function so update flag (gets used in resize)
        self.plotting_function_called = True

    def plot_tracks(self, tracks, mapping, uncertainty=False, resize=True,
                    particle=False, plot_history=False, ellipse_points=30,
                    track_label="Tracks", **kwargs):
        """
        Plots each track generated, generating a legend automatically. If 'uncertainty=True',
        error ellipses are plotted. Tracks are plotted as solid lines with point markers
        and default colours.

        Users can change linestyle, color, and marker using keyword arguments. Uncertainty metrics
        will also be plotted with the user defined colour and any changes will apply to all tracks.

        Parameters
        ----------
        tracks: Collection of :class '~Track'
            Collection of tracks which will be plotted. If not a collection, and instead a single
            :class:'~Track' type, the argument is modified to be a set to allow for iteration

        mapping: list
            List of items specifying the mapping of the position
            components of the state space
        uncertainty: bool
            If True, function plots uncertainty ellipses
        resize: bool
            If True, plotter will change bounds so that tracks are in view
        particle: bool
            If True, function plots particles
        plot_history: bool
            If true, plots all particles and uncertainty ellipses up to current time step
        ellipse_points: int
            Number of points for polygon approximating ellipse shape
        track_label: str
            Label to apply to all tracks for legend
        \\*\\*kwargs: dict
            Additional arguments to be passed to plot function. Defaults are ``linestyle="-"``,
            ``marker='s'`` for :class:`~.Update` and ``marker='o'`` for other states.

        Returns
        -------
        """

        if not isinstance(tracks, Collection) or isinstance(tracks, StateMutableSequence):
            tracks = {tracks}  # Make a set of length 1

        # So that we can plot tracks for both the current time and for some previous times,
        # we put plotting data for each track into a dictionary so that it can be easily
        # accessed later.
        data = [dict() for _ in tracks]

        for n, track in enumerate(tracks):  # sum up means - accounts for particle filter

            xydata = np.concatenate(
                [(getattr(state, 'mean', state.state_vector)[mapping, :])
                 for state in track],
                axis=1)

            # initialise arrays that go inside the dictionary
            data[n].update(x=xydata[0],
                           y=xydata[1],
                           time=np.array([0 for _ in range(len(track))], dtype=object),
                           time_str=np.array([0 for _ in range(len(track))], dtype=object),
                           type=np.array([0 for _ in range(len(track))], dtype=object))

            for k, state in enumerate(track):
                # fill the arrays here
                data[n]["time"][k] = state.timestamp
                data[n]["time_str"][k] = str(state.timestamp)
                data[n]["type"][k] = type(state).__name__

        trace_base = len(self.fig.data)  # number of traces

        # add dummy trace for legend for track

        track_kwargs = dict(x=[], y=[], mode="markers+lines", line=dict(color=self.colorway[2]),
                            legendgroup=track_label, legendrank=400, name=track_label,
                            showlegend=True)
        track_kwargs.update(kwargs)
        self.fig.add_trace(go.Scatter(track_kwargs))

        # and initialise traces for every track. Need to change a few kwargs:
        track_kwargs.update({'showlegend': False})

        for k, _ in enumerate(tracks):
            # update track colours
            track_kwargs.update({'line': dict(color=self.colorway[(k + 2) % len(self.colorway)])})
            track_kwargs.update(kwargs)
            self.fig.add_trace(go.Scatter(track_kwargs))

        for frame in self.fig.frames:
            # get current fig data and traces
            data_ = list(frame.data)
            traces_ = list(frame.traces)

            # convert string to datetime object
            frame_time = datetime.fromisoformat(frame.name)

            self.all_masks[frame_time] = dict()  # save mask for later use
            cutoff_time = (frame_time - self.time_window)
            # add blank data to ensure legend stays in place
            data_.append(go.Scatter(x=[-np.inf, np.inf], y=[-np.inf, np.inf]))
            traces_.append(trace_base)  # ensure data is added to correct trace

            for n, track in enumerate(tracks):

                # all track points that come at or before the frame time
                t_upper = [data[n]["time"] <= frame_time]
                # only select detections that come after the time cut-off
                t_lower = [data[n]["time"] >= cutoff_time]

                # put together
                mask = np.logical_and(t_upper, t_lower)

                # put into dictionary for later use
                if plot_history:
                    self.all_masks[frame_time][n] = np.logical_and(t_upper, t_lower)
                else:
                    self.all_masks[frame_time][n] = [data[n]["time"] == frame_time]

                # find x, y, time, and type
                track_x = data[n]["x"][tuple(mask)]
                # add np.inf to plot so that the traces are present for entire simulation
                track_x = np.append(track_x, [np.inf])

                # repeat for y
                track_y = data[n]["y"][tuple(mask)]
                track_y = np.append(track_y, [np.inf])
                track_type = data[n]["type"][tuple(mask)]
                times = data[n]["time_str"][tuple(mask)]

                data_.append(go.Scatter(x=track_x,  # plot track
                                        y=track_y,
                                        meta=track_type,
                                        customdata=times,
                                        hovertemplate='%{meta}' +
                                                      '<br>(%{x}, %{y})' +
                                                      '<br>Time: %{customdata}'))

                traces_.append(trace_base + n + 1)  # add to correct trace

                frame.data = data_
                frame.traces = traces_

        if resize:
            self._resize(data, "tracks")

        if uncertainty:  # plot ellipses

            uncertainty_kwargs = dict(x=[], y=[], legendgroup='Uncertainty', fill='toself',
                                      fillcolor=self.colorway[2],
                                      opacity=0.2, legendrank=500, name='Track<br>Uncertainty',
                                      hoverinfo='skip',
                                      mode='none', showlegend=True)
            uncertainty_kwargs.update(kwargs)

            # dummy trace for legend for uncertainty
            self.fig.add_trace(go.Scatter(uncertainty_kwargs))

            # and an uncertainty ellipse trace for each track
            uncertainty_kwargs.update({'showlegend': False})
            for k, _ in enumerate(tracks):
                uncertainty_kwargs.update(
                    {'fillcolor': self.colorway[(k + 2) % len(self.colorway)]})
                uncertainty_kwargs.update(kwargs)
                self.fig.add_trace(go.Scatter(uncertainty_kwargs))

            # following function finds uncertainty data points and plots them
            self._plot_particles_and_ellipses(tracks, mapping, resize, method="uncertainty")

        if particle:  # plot particles

            # initialise traces. One for legend and one per track

            particle_kwargs = dict(mode='markers', marker=dict(size=2, color=self.colorway[2]),
                                   opacity=0.4,
                                   hoverinfo='skip', legendgroup='particles', name='particles',
                                   legendrank=520, showlegend=True)
            # apply any keyword arguments
            particle_kwargs.update(kwargs)
            self.fig.add_trace(go.Scatter(particle_kwargs))  # legend trace

            particle_kwargs.update({"showlegend": False})

            for k, track in enumerate(tracks):  # trace for each track

                particle_kwargs.update(
                    {'marker': dict(size=2, color=self.colorway[(k + 2) % len(self.colorway)])})
                particle_kwargs.update(kwargs)
                self.fig.add_trace(go.Scatter(particle_kwargs))

            self._plot_particles_and_ellipses(tracks, mapping, resize, method="particles")

        # we have called a plotting function so update flag
        self.plotting_function_called = True

    def _plot_particles_and_ellipses(self, tracks, mapping, resize, method="uncertainty"):

        """
        The logic for plotting uncertainty ellipses and particles is nearly identical,
        so it is put into one function.

        Parameters
        ----------
        tracks: Collection of :class '~Track'
            Collection of tracks which will be plotted. If not a collection, and instead a single
            :class:'~Track' type, the argument is modified to be a set to allow for iteration
        mapping: list
            List of items specifying the mapping of the position components of the state space.
        method: str
            Can either be "uncertainty" or "particles". Depends on what the function is plotting.
        """

        data = [dict() for _ in tracks]
        trace_base = len(self.fig.data)
        for n, track in enumerate(tracks):

            # initialise arrays that store particle/ellipse for later plotting
            data[n].update(x=np.array([0 for _ in range(len(track))], dtype=object),
                           y=np.array([0 for _ in range(len(track))], dtype=object))

            for k, state in enumerate(track):

                # find data points
                if method == "uncertainty":

                    data_x, data_y = Plotterly._generate_ellipse_points(state, mapping)
                    data_x = list(data_x)
                    data_y = list(data_y)
                    data_x.append(np.nan)  # necessary to draw multiple ellipses at once
                    data_y.append(np.nan)
                    data[n]["x"][k] = data_x
                    data[n]["y"][k] = data_y

                elif method == "particles":

                    data_xy = state.state_vector[mapping[:2], :]
                    data[n]["x"][k] = data_xy[0]
                    data[n]["y"][k] = data_xy[1]

                else:
                    raise ValueError("Should be 'uncertainty' or 'particles'")

        for frame in self.fig.frames:

            frame_time = datetime.fromisoformat(frame.name)

            data_ = list(frame.data)  # current data in frame
            traces_ = list(frame.traces)  # current traces in frame

            data_.append(go.Scatter(x=[-np.inf], y=[np.inf]))  # add empty data for legend trace
            traces_.append(trace_base - len(tracks) - 1)  # ensure correct trace

            for n, track in enumerate(tracks):
                # now plot the data
                _x = list(chain(*data[n]["x"][tuple(self.all_masks[frame_time][n])]))
                _y = list(chain(*data[n]["y"][tuple(self.all_masks[frame_time][n])]))
                _x.append(np.inf)
                _y.append(np.inf)
                data_.append(go.Scatter(x=_x, y=_y))
                traces_.append(trace_base - len(tracks) + n)

            frame.data = data_
            frame.traces = traces_

        if resize:
            self._resize(data, type="particle_or_uncertainty")

    def plot_sensors(self, sensors, sensor_label="Sensors", resize=True, **kwargs):
        """Plots sensor(s)

        Plots sensors.  Users can change the color and marker of detections using keyword
        arguments. Default is a black 'x' marker. Currently only works for stationary
        sensors.

        Parameters
        ----------
        sensors : Collection of :class:`~.Sensor`
            Sensors to plot
        sensor_label: str
            Label to apply to all tracks for legend.
        \\*\\*kwargs: dict
            Additional arguments to be passed to scatter function for detections. Defaults are
            ``marker=dict(symbol='x', color='black')``.
        """
        if not isinstance(sensors, Collection):
            sensors = {sensors}

        # don't run any of this if there is no data input
        if sensors:
            trace_base = len(self.fig.data)  # number of traces currently in figure
            sensor_kwargs = dict(mode='markers', marker=dict(symbol='x', color='black'),
                                 legendgroup=sensor_label, legendrank=50,
                                 name=sensor_label, showlegend=True)
            sensor_kwargs.update(kwargs)

            self.fig.add_trace(go.Scatter(sensor_kwargs))  # initialises trace

            # sensor position
            sensor_xy = np.array([sensor.position[[0, 1], 0] for sensor in sensors])
            if resize:
                self._resize(sensors, "sensor")

            for frame in self.fig.frames:  # the plotting bit
                traces_ = list(frame.traces)
                data_ = list(frame.data)

                data_.append(go.Scatter(x=sensor_xy[:, 0], y=sensor_xy[:, 1]))
                traces_.append(trace_base)

                frame.traces = traces_
                frame.data = data_

        # we have called a plotting function so update flag (used in _resize)
        self.plotting_function_called = True
