import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pytest

from stonesoup.dataassociator.neighbour import NearestNeighbour
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.plotter import Plotter, Dimension, AnimatedPlotterly, AnimationPlotter, Plotterly, \
    PolarPlotterly
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.sensor.radar.radar import RadarElevationBearingRange
from stonesoup.types.detection import TrueDetection, Clutter
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.state import GaussianState, State
from stonesoup.types.track import Track
from stonesoup.updater.kalman import KalmanUpdater

# Setup simulation to test the plotter functionality
start_time = datetime.now()
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                          ConstantVelocity(0.005)])
truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=start_time)])
for k in range(1, 21):
    truth.append(GroundTruthState(
        transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=start_time+timedelta(seconds=k)))
timesteps = [start_time + timedelta(seconds=k) for k in range(1, 21)]

measurement_model = LinearGaussian(
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.array([[0.75, 0],
                          [0, 0.75]]))
true_measurements = []
for state in truth:
    measurement_set = set()
    # Generate actual detection from the state with a 1-p_d chance that no detection is received.
    measurement = measurement_model.function(state, noise=True)
    measurement_set.add(TrueDetection(state_vector=measurement,
                                      groundtruth_path=truth,
                                      timestamp=state.timestamp,
                                      measurement_model=measurement_model))

    true_measurements.append(measurement_set)

clutter_measurements = []
for state in truth:
    clutter_measurement_set = set()
    random_state = state.from_state(
        state=state,
        state_vector=np.random.uniform(-20, 20, size=state.state_vector.size)
    )
    measurement = measurement_model.function(random_state, noise=True)
    clutter_measurement_set.add(Clutter(state_vector=measurement,
                                        timestamp=state.timestamp,
                                        measurement_model=measurement_model))

    clutter_measurements.append(clutter_measurement_set)

all_measurements = [*true_measurements, *clutter_measurements]

predictor = KalmanPredictor(transition_model)
updater = KalmanUpdater(measurement_model)
hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(), missed_distance=3)
data_associator = NearestNeighbour(hypothesiser)

# Run Kalman filter with data association
# Create prior
prior = GaussianState([[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)
track = Track([prior])
for n, measurements in enumerate(true_measurements):
    hypotheses = data_associator.associate([track],
                                           measurements,
                                           start_time + timedelta(seconds=n))
    hypothesis = hypotheses[track]  # get the hypothesis for the specified track

    if hypothesis.measurement:
        post = updater.update(hypothesis)
        track.append(post)
    else:  # When data associator says no detections are good enough, we'll keep the prediction
        track.append(hypothesis.prediction)

sensor2d = RadarElevationBearingRange(
    position_mapping=(0, 2),
    noise_covar=np.array([[0, 0],
                          [0, 0]]),
    ndim_state=4,
    position=np.array([[10], [50]]))

sensor3d = RadarElevationBearingRange(
    position_mapping=(0, 2, 4),
    noise_covar=np.array([[0, 0, 0],
                          [0, 0, 0]]),
    ndim_state=6,
    position=np.array([[10], [50], [0]])
)


@pytest.fixture(autouse=True)
def close_figs():
    existing_figs = set(plt.get_fignums())
    yield None
    for fignum in set(plt.get_fignums()) - existing_figs:
        plt.close(fignum)


@pytest.fixture(scope="module")
def plotter_class(request):

    plotter_class = request.param
    assert plotter_class in {Plotter, Plotterly, AnimationPlotter,
                             PolarPlotterly, AnimatedPlotterly}

    def _generate_animated_plotterly(*args, **kwargs):
        return AnimatedPlotterly(*args, timesteps=timesteps, **kwargs)

    def _generate_plotter(*args, **kwargs):
        return plotter_class(*args, **kwargs)

    if plotter_class in {Plotter, Plotterly, AnimationPlotter, PolarPlotterly}:
        yield _generate_plotter
    elif plotter_class is AnimatedPlotterly:
        yield _generate_animated_plotterly
    else:
        raise ValueError("Invalid Plotter type.")


# Test functions
def test_dimension_inlist():  # ensure dimension type is in predefined enum list
    with pytest.raises(AttributeError):
        Plotter(dimension=Dimension.TESTERROR)


def test_particle_3d():  # warning should arise if particle is attempted in 3d mode
    plotter3 = Plotter(dimension=Dimension.THREE)

    with pytest.raises(NotImplementedError):
        plotter3.plot_tracks(track, [0, 1, 2], particle=True, uncertainty=False)


def test_plot_sensors():
    plotter3d = Plotter(Dimension.THREE)
    plotter3d.plot_sensors(sensor3d, marker='o', color='red')
    assert 'Sensors' in plotter3d.legend_dict


@pytest.mark.parametrize(
    "plotter_class",
    [Plotter, Plotterly, AnimationPlotter, PolarPlotterly, AnimatedPlotterly], indirect=True)
def test_empty_tracks(plotter_class):
    plotter = plotter_class()
    plotter.plot_tracks(set(), [0, 2])


def test_figsize():
    plotter_figsize_default = Plotter()
    plotter_figsize_different = Plotter(figsize=(20, 15))
    assert plotter_figsize_default.fig.get_figwidth() == 10
    assert plotter_figsize_default.fig.get_figheight() == 6
    assert plotter_figsize_different.fig.get_figwidth() == 20
    assert plotter_figsize_different.fig.get_figheight() == 15


def test_equal_3daxis():
    plotter_default = Plotter(dimension=Dimension.THREE)
    plotter_xy_default = Plotter(dimension=Dimension.THREE)
    plotter_xy = Plotter(dimension=Dimension.THREE)
    plotter_xyz = Plotter(dimension=Dimension.THREE)
    truths = GroundTruthPath(states=[State(state_vector=[-1000, -20, -3]),
                                     State(state_vector=[1000, 20, 3])])
    plotter_default.plot_ground_truths(truths, mapping=[0, 1, 2])
    plotter_xy_default.plot_ground_truths(truths, mapping=[0, 1, 2])
    plotter_xy.plot_ground_truths(truths, mapping=[1, 1, 2])
    plotter_xyz.plot_ground_truths(truths, mapping=[0, 1, 2])
    plotter_xy_default.set_equal_3daxis()
    plotter_xy.set_equal_3daxis([0, 1])
    plotter_xyz.set_equal_3daxis([0, 1, 2])
    plotters = [plotter_default, plotter_xy_default, plotter_xy, plotter_xyz]
    lengths = [3, 2, 2, 1]
    for plotter, length in zip(plotters, lengths):
        min_xyz = [0, 0, 0]
        max_xyz = [0, 0, 0]
        for i in range(3):
            for line in plotter.ax.lines:
                min_xyz[i] = np.min([min_xyz[i], *line.get_data_3d()[i]])
                max_xyz[i] = np.max([max_xyz[i], *line.get_data_3d()[i]])
        assert len(set(min_xyz)) == length
        assert len(set(max_xyz)) == length


def test_equal_3daxis_2d():
    plotter = Plotter(dimension=Dimension.TWO)
    truths = GroundTruthPath(states=[State(state_vector=[-1000, -20, -3]),
                                     State(state_vector=[1000, 20, 3])])
    plotter.plot_ground_truths(truths, mapping=[0, 1])
    plotter.set_equal_3daxis()


def test_plot_density_empty_state_sequences():
    plotter = Plotter()
    with pytest.raises(ValueError):
        plotter.plot_density([], index=None)


def test_plot_density_equal_x_y():
    plotter = Plotter()
    start_time = datetime.now()
    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0), ConstantVelocity(0)])
    truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], start_time)])
    for k in range(20):
        truth.append(GroundTruthState(
            transition_model.function(truth[k], noise=True,
                                      time_interval=timedelta(seconds=1)),
            timestamp=start_time + timedelta(seconds=k + 1)))
    with pytest.raises(ValueError):
        plotter.plot_density({truth}, index=None)


def test_plot_complex_uncertainty():
    plotter = Plotter()
    track = Track([
        GaussianState(
            state_vector=[0, 0],
            covar=[[10, -1], [1, 10]])
    ])
    with pytest.warns(UserWarning, match="Can not plot uncertainty for all states due to complex "
                                         "eigenvalues or eigenvectors"):

        plotter.plot_tracks(track, mapping=[0, 1], uncertainty=True)


def test_animation_plotter():
    animation_plotter = AnimationPlotter()
    animation_plotter.plot_ground_truths(truth, [0, 2])
    animation_plotter.plot_measurements(true_measurements, [0, 2])
    animation_plotter.run()

    animation_plotter_with_title = AnimationPlotter(title="Plot title")
    animation_plotter_with_title.plot_ground_truths(truth, [0, 2])
    animation_plotter_with_title.plot_tracks(track, [0, 2])
    animation_plotter_with_title.run()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            "Animation was deleted without rendering anything"
        )
        del animation_plotter
        del animation_plotter_with_title


def test_animated_plotterly():
    plotter = AnimatedPlotterly(timesteps)
    plotter.plot_ground_truths(truth, [0, 2])
    plotter.plot_measurements(true_measurements, [0, 2])
    plotter.plot_tracks(track, [0, 2], uncertainty=True, plot_history=True)


def test_animated_plotterly_empty():
    plotter = AnimatedPlotterly(timesteps)
    plotter.plot_ground_truths({}, [0, 2])
    plotter.plot_measurements({}, [0, 2])
    plotter.plot_tracks({}, [0, 2])
    plotter.plot_sensors({})


def test_animated_plotterly_sensor_plot():
    plotter = AnimatedPlotterly([start_time, start_time+timedelta(seconds=1)])
    plotter.plot_sensors(sensor2d)


def test_animated_plotterly_uneven_times():
    with pytest.warns(UserWarning, match="Timesteps are not equally spaced, so the passage of "
                                         "time is not linear"):
        AnimatedPlotterly([start_time,
                           start_time + timedelta(seconds=1),
                           start_time + timedelta(seconds=3)])


def test_plotterly_empty():
    plotter = Plotterly()
    plotter.plot_ground_truths(set(), [0, 2])
    plotter.plot_measurements(set(), [0, 2])
    plotter.plot_tracks(set(), [0, 2])
    with pytest.raises(TypeError):
        plotter.plot_tracks(set())
    with pytest.raises(ValueError):
        plotter.plot_tracks(set(), [])


def test_plotterly_1d():
    plotter1d = Plotterly(dimension=1)
    plotter1d.plot_ground_truths(truth, [0])
    plotter1d.plot_measurements(true_measurements, [0])
    plotter1d.plot_tracks(track, [0])

    # check that particle=True does not plot
    with pytest.raises(NotImplementedError):
        plotter1d.plot_tracks(track, [0], particle=True)

    # check that uncertainty=True does not plot
    with pytest.raises(NotImplementedError):
        plotter1d.plot_tracks(track, [0], uncertainty=True)


def test_plotterly_2d():
    plotter2d = Plotterly()
    plotter2d.plot_ground_truths(truth, [0, 2])
    plotter2d.plot_measurements(true_measurements, [0, 2])
    plotter2d.plot_tracks(track, [0, 2], uncertainty=True)
    plotter2d.plot_sensors(sensor2d)


def test_plotterly_3d():
    plotter3d = Plotterly(dimension=3)
    plotter3d.plot_ground_truths(truth, [0, 1, 2])
    plotter3d.plot_measurements(true_measurements, [0, 1, 2])
    plotter3d.plot_tracks(track, [0, 1, 2], uncertainty=True)

    with pytest.raises(NotImplementedError):
        plotter3d.plot_tracks(track, [0, 1, 2], particle=True)


@pytest.mark.parametrize("dim, mapping", [
    (1, [0, 1]),
    (1, [0, 1, 2]),
    (2, [0]),
    (2, [0, 1, 2]),
    (3, [0]),
    (3, [0, 1])])
def test_plotterly_wrong_dimension(dim, mapping):
    # ensure that plotter doesn't run for truth, measurements, and tracks
    # if dimension of those are not the same as the plotter's dimension
    plotter = Plotterly(dimension=dim)
    with pytest.raises(TypeError):
        plotter.plot_ground_truths(truth, mapping)

    with pytest.raises(TypeError):
        plotter.plot_measurements(true_measurements, mapping)

    with pytest.raises(TypeError):
        plotter.plot_tracks(track, mapping)


@pytest.mark.parametrize("labels", [
    None, ["Tracks"], ["Ground Truth", "Tracks"],
    ["Ground Truth", "Measurements", "Tracks"]])
def test_hide_plot(labels):
    plotter = Plotterly()
    plotter.plot_ground_truths(truth, [0, 1])
    plotter.plot_measurements(true_measurements, [0, 1])
    plotter.plot_tracks(track, [0, 1])

    plotter.hide_plot_traces(labels)

    hidden = 0
    showing = 0

    for fig_data in plotter.fig.data:
        if fig_data["visible"] == "legendonly":
            hidden += 1
        elif fig_data["visible"] is None:
            showing += 1

    if labels is None:
        assert hidden == 3
    else:
        assert hidden == len(labels)
    assert hidden + showing == 3


@pytest.mark.parametrize("labels", [
    None, ["Tracks"], ["Ground Truth", "Tracks"],
    ["Ground Truth", "Measurements", "Tracks"]])
def test_show_plot(labels):
    plotter = Plotterly()
    plotter.plot_ground_truths(truth, [0, 1])
    plotter.plot_measurements(true_measurements, [0, 1])
    plotter.plot_tracks(track, [0, 1])

    plotter.show_plot_traces(labels)

    showing = 0
    hidden = 0

    for fig_data in plotter.fig.data:
        if fig_data["visible"] == "legendonly":
            hidden += 1
        elif fig_data["visible"] is None:
            showing += 1

    if labels is None:
        assert showing == 3
    else:
        assert showing == len(labels)
    assert showing + hidden == 3


@pytest.mark.parametrize(
    "plotter_class",
    [Plotter, Plotterly, AnimationPlotter, PolarPlotterly, AnimatedPlotterly], indirect=True)
@pytest.mark.parametrize(
    "_measurements",
    [true_measurements, clutter_measurements, all_measurements,
     all_measurements[0]  # Tests a single measurement outside of a Collection should still run
     ])
def test_plotters_plot_measurements_2d(plotter_class, _measurements):
    plotter = plotter_class()
    plotter.plot_measurements(_measurements, [0, 2])


@pytest.mark.parametrize(
    "plotter_class",
    [Plotter, Plotterly, AnimationPlotter, PolarPlotterly, AnimatedPlotterly], indirect=True)
def test_plotters_plot_tracks(plotter_class):
    plotter = plotter_class()
    plotter.plot_tracks(track, [0, 2])


@pytest.mark.parametrize(
    "plotter_class",
    [Plotter,
     Plotterly,
     pytest.param(AnimationPlotter, marks=pytest.mark.xfail(raises=NotImplementedError)),
     pytest.param(PolarPlotterly, marks=pytest.mark.xfail(raises=NotImplementedError)),
     AnimatedPlotterly],
    indirect=True
)
def test_plotters_plot_track_uncertainty(plotter_class):
    plotter = plotter_class()
    plotter.plot_tracks(track, [0, 2], uncertainty=True)


@pytest.mark.xfail(raises=NotImplementedError)
@pytest.mark.parametrize(
    "plotter_class",
    [AnimationPlotter,
     PolarPlotterly]
)
def test_plotters_plot_track_particle(plotter_class):
    plotter = plotter_class()
    plotter.plot_tracks(track, [0, 2], particle=True)


@pytest.mark.parametrize(
    "plotter_class",
    [Plotter, Plotterly, AnimationPlotter, PolarPlotterly, AnimatedPlotterly], indirect=True)
def test_plotters_plot_truths(plotter_class):
    plotter = plotter_class()
    plotter.plot_ground_truths(truth, [0, 2])


@pytest.mark.parametrize(
    "plotter_class",
    [Plotter,
     Plotterly,
     pytest.param(AnimationPlotter, marks=pytest.mark.xfail(raises=NotImplementedError)),
     pytest.param(PolarPlotterly, marks=pytest.mark.xfail(raises=NotImplementedError)),
     AnimatedPlotterly], indirect=True
)
def test_plotters_plot_sensors(plotter_class):
    plotter = plotter_class()
    plotter.plot_sensors(sensor2d)


@pytest.mark.parametrize("plotter_class",
                         [Plotterly, PolarPlotterly, AnimatedPlotterly], indirect=True)
@pytest.mark.parametrize("_measurements, expected_labels",
                         [(true_measurements, {'Measurements'}),
                          (clutter_measurements, {'Measurements<br>(Clutter)'}),
                          (all_measurements, {'Measurements<br>(Detections)',
                                              'Measurements<br>(Clutter)'})
                          ])
def test_plotterlys_plot_measurements_label(plotter_class, _measurements, expected_labels):
    plotter = plotter_class()
    plotter.plot_measurements(_measurements, [0, 2])
    actual_labels = {fig_data.legendgroup for fig_data in plotter.fig.data}
    assert actual_labels == expected_labels


@pytest.mark.parametrize("_measurements, expected_labels",
                         [(true_measurements, {'Measurements'}),
                          (clutter_measurements, {'Measurements\n(Clutter)'}),
                          (all_measurements, {'Measurements\n(Detections)',
                                              'Measurements\n(Clutter)'})
                          ])
def test_plotter_plot_measurements_label(_measurements, expected_labels):
    plotter = Plotter()
    plotter.plot_measurements(_measurements, [0, 2])
    actual_labels = set(plotter.legend_dict.keys())
    assert actual_labels == expected_labels
