import warnings
from abc import ABC, abstractmethod
from itertools import chain
from typing import Collection, Iterable, Union

import numpy as np
from scipy.stats import kde
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.legend_handler import HandlerPatch
from scipy.integrate import quad
from scipy.optimize import brentq

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

from .types import detection
from .types.groundtruth import GroundTruthPath
from .types.state import State, StateMutableSequence
from .types.update import Update

from .models.base import LinearModel, Model

from enum import Enum


class Dimension(Enum):
    """Dimension Enum class for specifying plotting parameters in the Plotter class.
    Used to sanitize inputs for the dimension attribute of Plotter().

    Attributes
    ----------
    TWO: str
        Specifies 2D plotting for Plotter object
    THREE: str
        Specifies 3D plotting for Plotter object
    """
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
    def plot_sensors(self, sensors, sensor_label="Sensors", **kwargs):
        raise NotImplementedError

    def _conv_measurements(self, measurements, mapping, measurement_model=None):
        conv_detections = {}
        conv_clutter = {}
        for state in measurements:
            meas_model = state.measurement_model  # measurement_model from detections
            if meas_model is None:
                meas_model = measurement_model  # measurement_model from input

            if isinstance(meas_model, LinearModel):
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
        \\*\\*kwargs: dict
            Additional arguments to be passed to plot function. Default is ``linestyle="--"``.
        """

        truths_kwargs = dict(linestyle="--")
        truths_kwargs.update(kwargs)
        if not isinstance(truths, Collection) or isinstance(truths, StateMutableSequence):
            truths = {truths}  # Make a set of length 1

        for truth in truths:
            if self.dimension is Dimension.TWO:  # plots the ground truths in xy
                self.ax.plot([state.state_vector[mapping[0]] for state in truth],
                             [state.state_vector[mapping[1]] for state in truth],
                             **truths_kwargs)
            elif self.dimension is Dimension.THREE:  # plots the ground truths in xyz
                self.ax.plot3D([state.state_vector[mapping[0]] for state in truth],
                               [state.state_vector[mapping[1]] for state in truth],
                               [state.state_vector[mapping[2]] for state in truth],
                               **truths_kwargs)
            else:
                raise NotImplementedError('Unsupported dimension type for truth plotting')
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
        measurements : Collection of :class:`~.Detection`
            Detections which will be plotted. If measurements is a set of lists it is flattened.
        mapping: list
            List of items specifying the mapping of the position components of the state space.
        measurement_model : :class:`~.Model`, optional
            User-defined measurement model to be used in finding measurement state inverses if
            they cannot be found from the measurements themselves.
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
                                                                measurement_model)

        if plot_detections:
            detection_array = np.array(list(plot_detections.values()))
            # *detection_array.T unpacks detection_array by columns
            # (same as passing in detection_array[:,0], detection_array[:,1], etc...)
            self.ax.scatter(*detection_array.T, **measurement_kwargs)
            measurements_handle = Line2D([], [], linestyle='', **measurement_kwargs)

            # Generate legend items for measurements
            self.legend_dict[measurements_label] = measurements_handle

        if plot_clutter:
            clutter_array = np.array(list(plot_clutter.values()))
            self.ax.scatter(*clutter_array.T, color='y', marker='2')
            clutter_handle = Line2D([], [], linestyle='', marker='2', color='y')
            clutter_label = "Clutter"

            # Generate legend items for clutter
            self.legend_dict[clutter_label] = clutter_handle

        # Generate legend
        self.ax.legend(handles=self.legend_dict.values(), labels=self.legend_dict.keys())

    def plot_tracks(self, tracks, mapping, uncertainty=False, particle=False, track_label="Tracks",
                    err_freq=1, **kwargs):
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
        \\*\\*kwargs: dict
            Additional arguments to be passed to plot function. Defaults are ``linestyle="-"``,
            ``marker='s'`` for :class:`~.Update` and ``marker='o'`` for other states.
        """

        tracks_kwargs = dict(linestyle='-', marker="s", color=None)
        tracks_kwargs.update(kwargs)
        if not isinstance(tracks, Collection) or isinstance(tracks, StateMutableSequence):
            tracks = {tracks}  # Make a set of length 1

        # Plot tracks
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
            if not_update_indexes:
                self.ax.plot(
                    *data[:, not_update_indexes],
                    marker="o" if "marker" not in kwargs else kwargs['marker'],
                    color=plt.getp(line[0], 'color'))
            track_colors[track] = plt.getp(line[0], 'color')

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
                    for state in track:
                        w, v = np.linalg.eig(HH @ state.covar @ HH.T)
                        max_ind = np.argmax(w)
                        min_ind = np.argmin(w)
                        orient = np.arctan2(v[1, max_ind], v[0, max_ind])
                        ellipse = Ellipse(xy=state.mean[mapping[:2], 0],
                                          width=2 * np.sqrt(w[max_ind]),
                                          height=2 * np.sqrt(w[min_ind]),
                                          angle=np.rad2deg(orient), alpha=0.2,
                                          color=track_colors[track])
                        self.ax.add_artist(ellipse)

                # Generate legend items for uncertainty ellipses
                ellipse_handle = Ellipse((0.5, 0.5), 0.5, 0.5, alpha=0.2,
                                         color=tracks_kwargs['color'])
                ellipse_label = "Uncertainty"
                self.legend_dict[ellipse_label] = ellipse_handle
                # Generate legend
                self.ax.legend(handles=self.legend_dict.values(),
                               labels=self.legend_dict.keys(),
                               handler_map={Ellipse: _HandlerEllipse()})
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

                            self.ax.plot3D([xl+x_err, xl-x_err], [yl, yl], [zl, zl],
                                           marker="_", color=tracks_kwargs['color'])
                            self.ax.plot3D([xl, xl], [yl+y_err, yl-y_err], [zl, zl],
                                           marker="_", color=tracks_kwargs['color'])
                            self.ax.plot3D([xl, xl], [yl, yl], [zl+z_err, zl-z_err],
                                           marker="_", color=tracks_kwargs['color'])
                        check += 1

        if particle:
            if self.dimension is Dimension.TWO:
                # Plot particles
                for track in tracks:
                    for state in track:
                        data = state.state_vector[mapping[:2], :]
                        self.ax.plot(data[0], data[1], linestyle='', marker=".",
                                     markersize=1, alpha=0.5)

                # Generate legend items for particles
                particle_handle = Line2D([], [], linestyle='', color="black", marker='.',
                                         markersize=1)
                particle_label = "Particles"
                self.legend_dict[particle_label] = particle_handle
                # Generate legend
                self.ax.legend(handles=self.legend_dict.values(),
                               labels=self.legend_dict.keys())  # particle error legend
            else:
                raise NotImplementedError("""Particle plotting is not currently supported for
                                          3D visualization""")

        else:
            self.ax.legend(handles=self.legend_dict.values(), labels=self.legend_dict.keys())

    def plot_sensors(self, sensors, sensor_label="Sensors", **kwargs):
        """Plots sensor(s)

        Plots sensors.  Users can change the color and marker of detections using keyword
        arguments. Default is a black 'x' marker.

        Parameters
        ----------
        sensors : Collection of :class:`~.Sensor`
            Sensors to plot
        sensor_label: str
            Label to apply to all tracks for legend.
        \\*\\*kwargs: dict
            Additional arguments to be passed to plot function for detections. Defaults are
            ``marker='x'`` and ``color='black'``.
        """

        sensor_kwargs = dict(marker='x', color='black')
        sensor_kwargs.update(kwargs)

        if not isinstance(sensors, Collection):
            sensors = {sensors}  # Make a set of length 1

        for sensor in sensors:
            if self.dimension is Dimension.TWO:  # plots the sensors in xy
                self.ax.scatter(sensor.position[0],
                                sensor.position[1],
                                **sensor_kwargs)
            elif self.dimension is Dimension.THREE:  # plots the sensors in xyz
                self.ax.plot3D(sensor.position[0],
                               sensor.position[1],
                               sensor.position[2],
                               **sensor_kwargs)
            else:
                raise NotImplementedError('Unsupported dimension type for sensor plotting')
        self.legend_dict[sensor_label] = Line2D([], [], linestyle='', **sensor_kwargs)
        self.ax.legend(handles=self.legend_dict.values(), labels=self.legend_dict.keys())

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

    def plot_density(self, state_sequences: Iterable[StateMutableSequence],
                     index: Union[int, None] = -1,
                     mapping=(0, 2), n_bins=300, **kwargs):
        """

        Parameters
        ----------
        state_sequences : an iterable of :class:`~.StateMutableSequence`
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


class Plotterly(_Plotter):
    """Plotting class for building graphs of Stone Soup simulations using plotly

    A plotting class which is used to simplify the process of plotting ground truths,
    measurements, clutter and tracks. Tracks can be plotted with uncertainty ellipses or
    particles if required. Legends are automatically generated with each plot.
    Three dimensional plots can be created using the optional dimension parameter.

    Parameters
    ----------
    dimension: enum \'Dimension\'
        Optional parameter to specify 2D or 3D plotting. Currently only 2D plotting is
        supported.
    \\*\\*kwargs: dict
        Additional arguments to be passed to Figure.

    Attributes
    ----------
    fig: plotly.graph_objects.Figure
        Generated figure for graphs to be plotted on
    """
    def __init__(self, dimension=Dimension.TWO, **kwargs):
        if go is None:
            raise RuntimeError("Usage of Plotterly plotter requires installation of `plotly`")
        if isinstance(dimension, type(Dimension.TWO)):
            self.dimension = dimension
        else:
            raise TypeError("%s is an unsupported type for \'dimension\'; "
                            "expected type %s" % (type(dimension), type(Dimension.TWO)))
        if self.dimension != dimension.TWO:
            raise TypeError("Only 2D plotting currently supported")

        from plotly import colors
        layout_kwargs = dict(
            xaxis=dict(title=dict(text="<i>x</i>")),
            yaxis=dict(title=dict(text="<i>y</i>"), scaleanchor="x", scaleratio=1),
            colorway=colors.qualitative.Plotly,  # Needed to match colours later.
        )
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
        \\*\\*kwargs: dict
            Additional arguments to be passed to scatter function. Default is
            ``line=dict(dash="dash")``.
        """
        if not isinstance(truths, Collection) or isinstance(truths, StateMutableSequence):
            truths = {truths}

        truths_kwargs = dict(
            mode="lines", line=dict(dash="dash"), legendgroup=truths_label, legendrank=100,
            name=truths_label)
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
            self.fig.add_scatter(
                x=[state.state_vector[mapping[0]] for state in truth],
                y=[state.state_vector[mapping[1]] for state in truth],
                text=[self._format_state_text(state) for state in truth],
                **scatter_kwargs)

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
        measurements : Collection of :class:`~.Detection`
            Detections which will be plotted. If measurements is a set of lists it is flattened.
        mapping: list
            List of items specifying the mapping of the position components of the state space.
        measurement_model : :class:`~.Model`, optional
            User-defined measurement model to be used in finding measurement state inverses if
            they cannot be found from the measurements themselves.
        measurements_label : str
            Label for the measurements.  Default is "Measurements".
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
                                                                measurement_model)

        if plot_detections:
            name = measurements_label + "<br>(Detections)"
            measurement_kwargs = dict(
                mode='markers', marker=dict(color='#636EFA'),
                name=name, legendgroup=name, legendrank=200)
            measurement_kwargs.update(kwargs)
            if measurement_kwargs['legendgroup'] not in {trace.legendgroup
                                                         for trace in self.fig.data}:
                measurement_kwargs['showlegend'] = True
            else:
                measurement_kwargs['showlegend'] = False
            detection_array = np.array(list(plot_detections.values()))
            self.fig.add_scatter(
                x=detection_array[:, 0],
                y=detection_array[:, 1],
                text=[self._format_state_text(state) for state in plot_detections.keys()],
                **measurement_kwargs,
            )

        if plot_clutter:
            name = measurements_label + "<br>(Clutter)"
            measurement_kwargs = dict(
                mode='markers', marker=dict(symbol="star-triangle-up", color='#FECB52'),
                name=name, legendgroup=name, legendrank=210)
            measurement_kwargs.update(kwargs)
            if measurement_kwargs['legendgroup'] not in {trace.legendgroup
                                                         for trace in self.fig.data}:
                measurement_kwargs['showlegend'] = True
            else:
                measurement_kwargs['showlegend'] = False
            clutter_array = np.array(list(plot_clutter.values()))
            self.fig.add_scatter(
                x=clutter_array[:, 0],
                y=clutter_array[:, 1],
                text=[self._format_state_text(state) for state in plot_clutter.keys()],
                **measurement_kwargs,
            )

    def plot_tracks(self, tracks, mapping, uncertainty=False, particle=False, track_label="Tracks",
                    ellipse_points=30, **kwargs):
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
        \\*\\*kwargs: dict
            Additional arguments to be passed to scatter function. Defaults are
            ``marker=dict(symbol='square')`` for :class:`~.Update` and
            ``marker=dict(symbol='circle')`` for other states.
        """
        if not isinstance(tracks, Collection) or isinstance(tracks, StateMutableSequence):
            tracks = {tracks}  # Make a set of length 1

        # Plot tracks
        track_colors = {}
        track_kwargs = dict(mode='markers+lines', legendgroup=track_label, legendrank=300)
        track_kwargs.update(kwargs)
        add_legend = track_kwargs['legendgroup'] not in {trace.legendgroup
                                                         for trace in self.fig.data}
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

            self.fig.add_scatter(
                x=[getattr(state, 'mean', state.state_vector)[mapping[0]] for state in track],
                y=[getattr(state, 'mean', state.state_vector)[mapping[1]] for state in track],
                text=[self._format_state_text(state) for state in track],
                **scatter_kwargs)
            color = self.fig.data[-1].line.color
            if color is not None:
                track_colors[track] = color
            else:
                # This approach to getting colour isn't ideal, but should work in most cases...
                index = len(self.fig.data) - 1
                colorway = self.fig.layout.colorway
                max_index = len(colorway)
                track_colors[track] = colorway[index % max_index]

        if uncertainty:
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
        if particle:
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

        points = np.array([[a * np.sin(i), b * np.cos(i)] for i in points])
        points = rotational_matrix @ points.T
        return points + state.mean[mapping[:2], :]

    def plot_sensors(self, sensors, sensor_label="Sensors", **kwargs):
        """Plots sensor(s)

        Plots sensors.  Users can change the color and marker of detections using keyword
        arguments. Default is a black 'x' marker.

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

        sensor_kwargs = dict(mode='markers', marker=dict(symbol='x', color='black'),
                             legendgroup=sensor_label, legendrank=50)
        sensor_kwargs.update(kwargs)

        sensor_kwargs['name'] = sensor_label
        if sensor_kwargs['legendgroup'] not in {trace.legendgroup
                                                for trace in self.fig.data}:
            sensor_kwargs['showlegend'] = True
        else:
            sensor_kwargs['showlegend'] = True

        sensor_xy = np.array([sensor.position[[0, 1], 0] for sensor in sensors])
        self.fig.add_scatter(x=sensor_xy[:, 0], y=sensor_xy[:, 1], **sensor_kwargs)
