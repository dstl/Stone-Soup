"""Qudaratic Distance / Mean Quadratic Error tests."""
from datetime import datetime, timedelta
import numpy as np
from ordered_set import OrderedSet

from ..quadraticdistance import QuadraticDistance, MeanQuadraticError

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel
from stonesoup.models.transition.linear import ConstantVelocity
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.types.state import TaggedWeightedGaussianState
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.track import Track
from stonesoup.types.array import CovarianceMatrix
from stonesoup.metricgenerator.manager import MultiManager
from stonesoup.hypothesiser.gaussianmixture import GaussianMixtureHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.types.angle import Angle
from stonesoup.sensor.radar.radar import RadarRotatingBearingRange
from stonesoup.types.state import StateVector
from stonesoup.updater.pointprocess import PHDUpdater
from stonesoup.mixturereducer.gaussianmixture import GaussianMixtureReducer


def test_quadratic_distance():

    start_time = datetime.now()

    # test truths
    truths = OrderedSet()

    num_init_targs = 5

    for i in range(num_init_targs):
        x, y = np.random.uniform(-30, 30, 2)  # Range [-30, 30] for x and y
        x_vel, y_vel = (np.random.rand(2)) * 2 - 1  # Range [-1, 1] for x and y velocity
        state = GroundTruthState([x, x_vel, y, y_vel], timestamp=start_time)
        truth = GroundTruthPath([state])
        truths.add(truth)

    # test tracks A & B
    covar = CovarianceMatrix(np.diag([1, 0.5, 1, 0.5]))
    tracksA = set()
    tracksB = set()
    for truth in truths[0]:
        new_track_A = TaggedWeightedGaussianState(
            state_vector=truth.state_vector,
            covar=covar**2,
            weight=0.9,
            tag='birth',
            timestamp=start_time)

        new_track_B = TaggedWeightedGaussianState(
            state_vector=0.5*truth.state_vector,
            covar=2*covar**2,
            weight=0.9,
            tag='birth',
            timestamp=start_time)

        tracksA.add(Track([new_track_A]))
        tracksB.add(Track([new_track_B]))

    # test means A & B
    meansA = set()
    for track in tracksA:
        new_mean = []
        for state in track:
            new_mean.append(State(state_vector=state.state_vector.copy(),
                                timestamp=state.timestamp))
        meansA.add(Track(new_mean))

    meansB = set()
    for track in tracksB:
        new_mean = []
        for state in track:
            new_mean.append(State(state_vector=state.state_vector.copy(),
                                timestamp=state.timestamp))
        meansB.add(Track(new_mean))

    # metric initialisation
    kernel_cov = 100 * np.eye(4)

    # create a metric generator for each test
    # equal point sets
    quaderr_equal_pp = QuadraticDistance(state_dim=4,
                                         kernel='Gaussian',
                                         kernel_parameters={'covariance': kernel_cov},
                                         generator_name='equal points',
                                         tracks_key='truths', truths_key='truths')

    # equal track sets
    quaderr_equal_tt = QuadraticDistance(state_dim=4,
                                         kernel='Gaussian',
                                         kernel_parameters={'covariance': kernel_cov},
                                         generator_name='equal tracks',
                                         tracks_key='tracks1', truths_key='tracks1')

    # inequal track sets
    quaderr_inequal_tt = QuadraticDistance(state_dim=4,
                                           kernel='Gaussian',
                                           kernel_parameters={'covariance': kernel_cov},
                                           generator_name='inequal tracks',
                                           tracks_key='tracks2', truths_key='tracks1')

    # point - track
    quaderr_pt = QuadraticDistance(state_dim=4,
                                   kernel='Gaussian',
                                   kernel_parameters={'covariance': kernel_cov},
                                   generator_name='point - track',
                                   tracks_key='tracks1', truths_key='truths')

    # track - point
    quaderr_tp = QuadraticDistance(state_dim=4,
                                   kernel='Gaussian',
                                   kernel_parameters={'covariance': kernel_cov},
                                   generator_name='track - point',
                                   tracks_key='truths', truths_key='tracks1')

    # truths - means
    quaderr_truths_means = QuadraticDistance(state_dim=4,
                                             kernel='Gaussian',
                                             kernel_parameters={'covariance': kernel_cov},
                                             generator_name='truths - means',
                                             tracks_key='means1', truths_key='truths')

    # alt truths - means
    quaderr_alt_truths_means = QuadraticDistance(state_dim=4,
                                                 kernel='Gaussian',
                                                 kernel_parameters={'covariance': kernel_cov},
                                                 generator_name='alt truths - means',
                                                 tracks_key='means2', truths_key='truths')

    # means - means
    quaderr_means_means = QuadraticDistance(state_dim=4,
                                            kernel='Gaussian',
                                            kernel_parameters={'covariance': kernel_cov},
                                            generator_name='means - means',
                                            tracks_key='means2', truths_key='means1')

    manager = MultiManager([quaderr_equal_pp,
                            quaderr_equal_tt,
                            quaderr_inequal_tt,
                            quaderr_pt,
                            quaderr_tp,
                            quaderr_truths_means,
                            quaderr_alt_truths_means,
                            quaderr_means_means])

    manager.add_data({'truths': truths,
                         'means1': meansA,
                         'tracks1': tracksA,
                         'means2': meansB,
                         'tracks2': tracksB}, overwrite=False)
                        
    metrics = manager.generate_metrics()
    
    
    assert (metrics['equal points']['Quadratic Distance'].value
            == metrics['equal tracks']['Quadratic Distance'].value
            == 0)
    
    assert (metrics['point - track']['Quadratic Distance'].value
            == metrics['track - point']['Quadratic Distance'].value)
    
    assert (not (metrics['truths - means']['Quadratic Distance'].value
                 == metrics['alt truths - means']['Quadratic Distance'].value))
    
    assert (metrics['inequal tracks']['Quadratic Distance'].value > 0)
    
    assert (metrics['means - means']['Quadratic Distance'].value
            > metrics['inequal tracks']['Quadratic Distance'].value)

def test_mean_quadratic_error():

    start_time = datetime.now()

    transition_model = CombinedLinearGaussianTransitionModel((ConstantVelocity(0.3),
                                                              ConstantVelocity(0.3)))
    # test truths
    # timestep 1
    truths = OrderedSet()
    num_init_targs = 5
    for i in range(num_init_targs):
        x, y = np.random.uniform(-30, 30, 2)  # Range [-30, 30] for x and y
        x_vel, y_vel = (np.random.rand(2)) * 2 - 1  # Range [-1, 1] for x and y velocity
        state = GroundTruthState([x, x_vel, y, y_vel], timestamp=start_time)
        truth = GroundTruthPath([state])
        truths.add(truth)

    # timestep 2

    timestep = start_time + timedelta(seconds=1)

    # Update existing truths
    for truth in truths:
        prev_state = truth[-1]  # Last state in the path
        new_state_vector = transition_model.function(
            prev_state, noise=True, time_interval=timedelta(seconds=1))
        new_state = GroundTruthState(new_state_vector, timestamp=timestep)
        truth.append(new_state)

    # test prior tracks
    covar = CovarianceMatrix(np.diag([1, 0.5, 1, 0.5]))
    prior_tracks = set()
    for truth in truths[0]:
        new_track = TaggedWeightedGaussianState(
            state_vector=0.5 * truth.state_vector,
            covar=covar ** 2,
            weight=0.9,
            tag='birth',
            timestamp=start_time)

        prior_tracks.add(Track([new_track]))

    prior_states= set([track[-1] for track in prior_tracks])

    # test measurements
    measurements = set()

    sens_range = 1000
    sens_fov = np.radians(360)
    probability_detection = 0.99
    clutter_rate = 0
    sens_res = Angle(np.radians(90))
    sens_noise = np.array([[np.radians(0.01) ** 2, 0],
                        [0, 0.1 ** 2]])
    sens_rpm = 120
    surveillance_area = (0.5 * sens_range ** 2 * sens_fov)

    sensor = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=sens_noise,
        ndim_state=4,
        position=np.array([[0], [0]]),
        rpm=sens_rpm,
        fov_angle=sens_fov,
        dwell_centre=StateVector([0.0]),
        max_range=sens_range,
        resolution=sens_res
    )

    sensor.act(start_time + timedelta(seconds=1))
    for truth in truths:
        measurements |= sensor.measure(
            OrderedSet( gt for gt in
        truth[start_time + timedelta(seconds=1)]
        for truth in truths
    ),
    noise=True,
)

    # hypotheses
    hypotheses = []

    death_probability = 0.01

    kalman_predictor = KalmanPredictor(transition_model)

    extended_kalman_updater = ExtendedKalmanUpdater(measurement_model=None)

    base_hypothesiser = DistanceHypothesiser(kalman_predictor,
                                             extended_kalman_updater,
                                             Mahalanobis(),
                                             missed_distance=10)

    hypothesiser = GaussianMixtureHypothesiser(base_hypothesiser,
                                               order_by_detection=True)

    hypothesis = hypothesiser.hypothesise(prior_states,
                                        measurements,
                                        timestamp=start_time + timedelta(seconds=1),
                                        # keep our hypotheses ordered by detection, not by track
                                        order_by_detection=True)

    hypotheses.append(Track(hypothesis))

    # posterior track
    clutter_spatial_density = clutter_rate / surveillance_area

    updater = PHDUpdater(extended_kalman_updater,
                         clutter_spatial_density=clutter_spatial_density,
                         prob_detection=probability_detection,
                         prob_survival=1 - death_probability)

    merge_threshold = 1
    prune_threshold = 1E-10

    reducer = GaussianMixtureReducer(prune_threshold=prune_threshold,
                                     pruning=False,
                                     merge_threshold=merge_threshold,
                                     merging=False)

    updated_states = updater.update(hypothesis)

    reduced_states = set(reducer.reduce(updated_states))

    posterior_tracks = set()
    for reduced_state in reduced_states:
        posterior_tracks.add(Track(reduced_state))

    # posterior means
    means = set()
    for track in posterior_tracks:
        new_mean = []
        for state in track:
            new_mean.append(State(state_vector=state.state_vector.copy(),
                                timestamp=state.timestamp))
        means.add(Track(new_mean))

    # metric initialisation
    kernel_cov = 100 * np.eye(4)

    filter_data_dict = {'filter model': 'GMPHD',
                    'state dimension': 4,
                    'detection probability': probability_detection,
                    'surveillance area': surveillance_area,
                    'survival probability': 1 - death_probability,
                    'clutter rate': clutter_rate,
                    'predictor': kalman_predictor,
                    'updater': extended_kalman_updater}

    # create a metric generator for each test
    # test reference for (sqrt) bias term
    quaderr_points = QuadraticDistance(state_dim=4,
                                    kernel='Gaussian',
                                kernel_parameters={'covariance': kernel_cov},
                                generator_name='qd truths - posterior means',
                                tracks_key='means', truths_key='truths')

    # no covariance test
    mquaderr_points = MeanQuadraticError(state_dim=4,
                                        filter_data=filter_data_dict,
                                                kernel='Gaussian',
                                                kernel_parameters={'covariance': kernel_cov},
                                                generator_name='mqe truths - posterior means',
                                                tracks_key='means',
                                                hypotheses_key='hypotheses',
                                                truths_key='truths')

    # with added posterior covariance term
    mquaderr = MeanQuadraticError(state_dim=4,
                                filter_data=filter_data_dict,
                                                kernel='Gaussian',
                                                kernel_parameters={'covariance': kernel_cov},
                                                generator_name='truths - posterior intensity',
                                                tracks_key='tracks',
                                                hypotheses_key='hypotheses',
                                                truths_key='truths')

    # incorrect arguement order
    mquaderr_swap = MeanQuadraticError(state_dim=4,
                                    filter_data=filter_data_dict,
                                        kernel='Gaussian',
                                        kernel_parameters={'covariance': kernel_cov},
                                        generator_name='posterior intensity - truths',
                                        tracks_key='truths',
                                        hypotheses_key='hypotheses',
                                        truths_key='tracks')

    manager = MultiManager([mquaderr,
                            mquaderr_points,
                            quaderr_points,
                            mquaderr_swap])

    manager.add_data({'truths': truths,
                            'means': means,
                            'tracks': posterior_tracks,
                            'hypotheses': hypotheses
                            }, overwrite=False)

    metrics = manager.generate_metrics()

    assert (np.sqrt(metrics['mqe truths - posterior means']['MQE values'].value[0].value)
            == metrics['qd truths - posterior means']['Quadratic distances'].value[1].value)
    
    assert (metrics['truths - posterior intensity']['MQE values'].value[0].value
            > metrics['mqe truths - posterior means']['MQE values'].value[0].value)
    
    assert (not (metrics['truths - posterior intensity']['MQE values'].value[0].value
                 == metrics['posterior intensity - truths']['MQE values'].value[0].value))
