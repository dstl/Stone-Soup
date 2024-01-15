import numpy as np
from stonesoup.plotter import Plotter, Dimension, AnimatedPlotterly, AnimationPlotter, Plotterly
import pytest
import matplotlib.pyplot as plt

# Setup simulation to test the plotter functionality
from datetime import datetime
from datetime import timedelta

from stonesoup.types.detection import TrueDetection
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.sensor.radar.radar import RadarElevationBearingRange

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater

from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis

from stonesoup.dataassociator.neighbour import NearestNeighbour
from stonesoup.types.state import GaussianState, State

from stonesoup.types.track import Track

start_time = datetime.now()
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                          ConstantVelocity(0.005)])
truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=start_time)])
for k in range(1, 21):
    truth.append(GroundTruthState(
        transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=start_time+timedelta(seconds=k)))
timesteps = [start_time + timedelta(seconds=k) for k in range(1, 21)]
prob_det = 0.5

measurement_model = LinearGaussian(
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.array([[0.75, 0],
                          [0, 0.75]]))
all_measurements = []
for state in truth:
    measurement_set = set()
    # Generate actual detection from the state with a 1-p_d chance that no detection is received.
    if np.random.rand() <= prob_det:
        measurement = measurement_model.function(state, noise=True)
        measurement_set.add(TrueDetection(state_vector=measurement,
                                          groundtruth_path=truth,
                                          timestamp=state.timestamp,
                                          measurement_model=measurement_model))

    all_measurements.append(measurement_set)

predictor = KalmanPredictor(transition_model)
updater = KalmanUpdater(measurement_model)
hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(), missed_distance=3)
data_associator = NearestNeighbour(hypothesiser)

# Run Kalman filter with data association
# Create prior
prior = GaussianState([[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)
track = Track([prior])
for n, measurements in enumerate(all_measurements):
    hypotheses = data_associator.associate([track],
                                           measurements,
                                           start_time + timedelta(seconds=n))
    hypothesis = hypotheses[track]  # get the hypothesis for the specified track

    if hypothesis.measurement:
        post = updater.update(hypothesis)
        track.append(post)
    else:  # When data associator says no detections are good enough, we'll keep the prediction
        track.append(hypothesis.prediction)

plotter = Plotter()
# Test functions


def test_dimension_raise():
    with pytest.raises(ValueError):
        Plotter(dimension=1)  # expected to raise ValueError


def test_dimension_inlist():  # ensure dimension type is in predefined enum list
    with pytest.raises(AttributeError):
        Plotter(dimension=Dimension.TESTERROR)


def test_measurements_legend():
    plotter.plot_measurements(all_measurements, [0, 2])  # Measurements entry in legend dict
    plt.close()
    assert 'Measurements' in plotter.legend_dict


def test_measurement_clutter():  # no clutter should be plotted
    plotter.plot_measurements(all_measurements, [0, 2])
    plt.close()
    assert 'Clutter' not in plotter.legend_dict


def test_single_measurement():  # A single measurement outside of a Collection should still run
    plotter.plot_measurements(all_measurements[0], [0, 2])
    plt.close()


def test_particle_3d():  # warning should arise if particle is attempted in 3d mode
    plotter3 = Plotter(dimension=Dimension.THREE)

    with pytest.raises(NotImplementedError):
        plotter3.plot_tracks(track, [0, 1, 2], particle=True, uncertainty=False)


def test_plot_sensors():
    plotter3d = Plotter(Dimension.THREE)
    sensor = RadarElevationBearingRange(
        position_mapping=(0, 2, 4),
        noise_covar=np.array([[0, 0, 0],
                              [0, 0, 0]]),
        ndim_state=6,
        position=np.array([[10], [50], [0]])
    )
    plotter3d.plot_sensors(sensor, marker='o', color='red')
    plt.close()
    assert 'Sensors' in plotter3d.legend_dict


def test_empty_tracks():
    plotter.plot_tracks(set(), [0, 2])
    plt.close()


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
    for plotter, l in zip(plotters, lengths):
        min_xyz = [0, 0, 0]
        max_xyz = [0, 0, 0]
        for i in range(3):
            for line in plotter.ax.lines:
                min_xyz[i] = np.min([min_xyz[i], *line.get_data_3d()[i]])
                max_xyz[i] = np.max([max_xyz[i], *line.get_data_3d()[i]])
        assert len(set(min_xyz)) == l
        assert len(set(max_xyz)) == l


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
                                         "eignevalues or eigenvectors"):

        plotter.plot_tracks(track, mapping=[0, 1], uncertainty=True)


def test_animation_plotter():
    animation_plotter = AnimationPlotter()
    animation_plotter.plot_ground_truths(truth, [0, 2])
    animation_plotter.plot_measurements(all_measurements, [0, 2])
    animation_plotter.run()

    animation_plotter_with_title = AnimationPlotter(title="Plot title")
    animation_plotter_with_title.plot_ground_truths(truth, [0, 2])
    animation_plotter_with_title.plot_tracks(track, [0, 2])
    animation_plotter_with_title.run()


def test_animated_plotterly():
    plotter = AnimatedPlotterly(timesteps)
    plotter.plot_ground_truths(truth, [0, 2])
    plotter.plot_measurements(all_measurements, [0, 2])
    plotter.plot_tracks(track, [0, 2], uncertainty=True, plot_history=True)


def test_animated_plotterly_empty():
    plotter = AnimatedPlotterly(timesteps)
    plotter.plot_ground_truths({}, [0, 2])
    plotter.plot_measurements({}, [0, 2])
    plotter.plot_tracks({}, [0, 2])
    plotter.plot_sensors({})


def test_animated_plotterly_sensor_plot():
    plotter = AnimatedPlotterly([start_time, start_time+timedelta(seconds=1)])
    sensor = RadarElevationBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[0, 0],
                              [0, 0]]),
        ndim_state=4,
        position=np.array([[10], [50]]))
    plotter.plot_sensors(sensor)


def test_animated_plotterly_uneven_times():
    with pytest.warns(UserWarning, match="Timesteps are not equally spaced, so the passage of "
                                         "time is not linear"):
        AnimatedPlotterly([start_time,
                           start_time + timedelta(seconds=1),
                           start_time + timedelta(seconds=3)])


def test_plotterly_empty():
    plotter = Plotterly()
    plotter.plot_ground_truths({}, [0, 2])
    plotter.plot_measurements({}, [0, 2])
    plotter.plot_tracks({}, [0, 2])


def test_plotterly_dimension():

    Plotterly(Dimension.TWO)  # default
    Plotterly(2)  # check that ints are passed through
    with pytest.raises(TypeError):
        Plotterly(dimension=3)

    with pytest.raises(TypeError):
        Plotterly(dimension=Dimension.THREE)


def test_plotterly():
    plotter = Plotterly()
    plotter.plot_ground_truths(truth, [0, 2])
    plotter.plot_measurements(all_measurements, [0, 2])
    plotter.plot_tracks(track, [0, 2], uncertainty=True)
