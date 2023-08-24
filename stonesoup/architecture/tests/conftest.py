import pytest


import numpy as np
import random
from ordered_set import OrderedSet
from datetime import datetime, timedelta
import copy

from ..edge import Edge, DataPiece, Edges
from ..node import Node, RepeaterNode, SensorNode, FusionNode, SensorFusionNode
from ...types.track import Track
from ...sensor.categorical import HMMSensor
from ...models.measurement.categorical import MarkovianMeasurementModel
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from ordered_set import OrderedSet
from stonesoup.types.state import StateVector
from stonesoup.sensor.radar.radar import RadarRotatingBearingRange
from stonesoup.types.angle import Angle
from stonesoup.architecture import Architecture, NetworkArchitecture, InformationArchitecture, \
    CombinedArchitecture
from stonesoup.architecture.node import FusionNode, RepeaterNode, SensorNode, SensorFusionNode
from stonesoup.architecture.edge import Edge, Edges
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.types.state import GaussianState
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.tracker.simple import MultiTargetTracker
from stonesoup.architecture.edge import FusionQueue
from stonesoup.updater.wrapper import DetectionAndTrackSwitchingUpdater
from stonesoup.updater.chernoff import ChernoffUpdater
from stonesoup.feeder.track import Tracks2GaussianDetectionFeeder
from ...types.hypothesis import Hypothesis


@pytest.fixture
def edges():
    edge_a = Edge((Node(label="edge_a sender"), Node(label="edge_a recipient")))
    edge_b = Edge((Node(label="edge_b sender"), Node(label="edge_b recipient")))
    edge_c = Edge((Node(label="edge_c sender"), Node(label="edge_c recipient")))
    return {'a': edge_a, 'b': edge_b, 'c': edge_c}


@pytest.fixture
def nodes():
    E = np.array([[0.8, 0.1],  # P(small | bike), P(small | car)
                  [0.19, 0.3],  # P(medium | bike), P(medium | car)
                  [0.01, 0.6]])  # P(large | bike), P(large | car)

    model = MarkovianMeasurementModel(emission_matrix=E,
                                      measurement_categories=['small', 'medium', 'large'])

    hmm_sensor = HMMSensor(measurement_model=model)

    node_a = Node(label="node a")
    node_b = Node(label="node b")
    sensornode_1 = SensorNode(sensor=hmm_sensor, label='s1')
    sensornode_2 = SensorNode(sensor=hmm_sensor, label='s2')
    sensornode_3 = SensorNode(sensor=hmm_sensor, label='s3')
    sensornode_4 = SensorNode(sensor=hmm_sensor, label='s4')
    sensornode_5 = SensorNode(sensor=hmm_sensor, label='s5')
    sensornode_6 = SensorNode(sensor=hmm_sensor, label='s6')
    sensornode_7 = SensorNode(sensor=hmm_sensor, label='s7')
    sensornode_8 = SensorNode(sensor=hmm_sensor, label='s8')
    pnode_1 = SensorNode(sensor=hmm_sensor, label='p1', position=(0, 0))
    pnode_2 = SensorNode(sensor=hmm_sensor, label='p2', position=(-1, -1))
    pnode_3 = SensorNode(sensor=hmm_sensor, label='p3', position=(1, -1))

    return {"a": node_a, "b": node_b, "s1": sensornode_1, "s2": sensornode_2, "s3": sensornode_3,
            "s4": sensornode_4, "s5": sensornode_5, "s6": sensornode_6, "s7": sensornode_7,
            "s8": sensornode_8, "p1": pnode_1, "p2": pnode_2, "p3": pnode_3}


@pytest.fixture
def data_pieces(times, nodes):
    data_piece_a = DataPiece(node=nodes['a'], originator=nodes['a'],
                             data=Track([]), time_arrived=times['a'])
    data_piece_b = DataPiece(node=nodes['a'], originator=nodes['b'],
                             data=Track([]), time_arrived=times['b'])
    data_piece_fail = DataPiece(node=nodes['a'], originator=nodes['b'],
                                data="Not a compatible data type", time_arrived=times['b'])
    data_piece_hyp = DataPiece(node=nodes['a'], originator=nodes['b'],
                               data=Hypothesis(), time_arrived=times['b'])
    return {'a': data_piece_a, 'b': data_piece_b, 'fail': data_piece_fail, 'hyp': data_piece_hyp}


@pytest.fixture
def times():
    time_a = datetime.strptime("23/08/2023 13:36:00", "%d/%m/%Y %H:%M:%S")
    time_b = datetime.strptime("23/08/2023 13:37:00", "%d/%m/%Y %H:%M:%S")
    start_time = datetime.strptime("25/12/1306 23:47:00", "%d/%m/%Y %H:%M:%S")
    return {'a': time_a, 'b': time_b, 'start': start_time}


@pytest.fixture
def transition_model():
    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                              ConstantVelocity(0.005)])
    return transition_model


@pytest.fixture
def ground_truths(transition_model, times):
    start_time = times["start"]
    yps = range(0, 100, 10)  # y value for prior state
    truths = OrderedSet()
    ntruths = 3  # number of ground truths in simulation
    time_max = 60  # timestamps the simulation is observed over
    timesteps = [start_time + timedelta(seconds=k) for k in range(time_max)]

    xdirection = 1
    ydirection = 1

    # Generate ground truths
    for j in range(0, ntruths):
        truth = GroundTruthPath([GroundTruthState([0, xdirection, yps[j], ydirection],
                                                  timestamp=timesteps[0])], id=f"id{j}")

        for k in range(1, time_max):
            truth.append(
                GroundTruthState(transition_model.function(truth[k - 1], noise=True,
                                                           time_interval=timedelta(seconds=1)),
                                 timestamp=timesteps[k]))
        truths.add(truth)

        xdirection *= -1
        if j % 2 == 0:
            ydirection *= -1

    return truths


@pytest.fixture
def radar_sensors(times):
    start_time = times["start"]
    total_no_sensors = 5
    sensor_set = OrderedSet()
    for n in range(0, total_no_sensors):
        sensor = RadarRotatingBearingRange(
            position_mapping=(0, 2),
            noise_covar=np.array([[np.radians(0.5) ** 2, 0],
                                  [0, 1 ** 2]]),
            ndim_state=4,
            position=np.array([[10], [n * 20 - 40]]),
            rpm=60,
            fov_angle=np.radians(360),
            dwell_centre=StateVector([0.0]),
            max_range=np.inf,
            resolutions={'dwell_centre': Angle(np.radians(30))}
        )
        sensor_set.add(sensor)
    for sensor in sensor_set:
        sensor.timestamp = start_time

    return sensor_set


@pytest.fixture
def predictor():
    predictor = KalmanPredictor(transition_model)
    return predictor


@pytest.fixture
def updater(transition_model):
    updater = ExtendedKalmanUpdater(measurement_model=None)
    return updater


@pytest.fixture
def hypothesiser(predictor, updater):
    hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(),
                                        missed_distance=5)
    return hypothesiser


@pytest.fixture
def data_associator(hypothesiser):
    data_associator = GNNWith2DAssignment(hypothesiser)
    return data_associator


@pytest.fixture
def deleter(hypothesiser):
    deleter = CovarianceBasedDeleter(covar_trace_thresh=7)
    return deleter


@pytest.fixture
def initiator():
    initiator = MultiMeasurementInitiator(
        prior_state=GaussianState([[0], [0], [0], [0]], np.diag([0, 1, 0, 1])),
        measurement_model=None,
        deleter=deleter,
        data_associator=data_associator,
        updater=updater,
        min_points=2,
    )
    return initiator


@pytest.fixture
def tracker():
    tracker = MultiTargetTracker(initiator, deleter, None, data_associator, updater)
    return tracker


@pytest.fixture
def track_updater():
    track_updater = ChernoffUpdater(None)
    return track_updater


@pytest.fixture
def detection_updater():
    detection_updater = ExtendedKalmanUpdater(None)
    return detection_updater


@pytest.fixture
def detection_track_updater():
    detection_track_updater = DetectionAndTrackSwitchingUpdater(None, detection_updater,
                                                                track_updater)
    return detection_track_updater


@pytest.fixture
def fusion_queue():
    fq = FusionQueue()
    return fq


@pytest.fixture
def track_tracker():
    track_tracker = MultiTargetTracker(
        initiator, deleter, Tracks2GaussianDetectionFeeder(fusion_queue), data_associator,
        detection_track_updater)
    return track_tracker


@pytest.fixture
def radar_nodes(radar_sensors, fusion_queue):
    sensor_set = radar_sensors
    node_A = SensorNode(sensor=sensor_set[0])
    node_B = SensorNode(sensor=sensor_set[2])

    node_C_tracker = copy.deepcopy(tracker)
    node_C_tracker.detector = FusionQueue()
    node_C = FusionNode(tracker=node_C_tracker, fusion_queue=node_C_tracker.detector, latency=0)

    node_D = SensorNode(sensor=sensor_set[1])
    node_E = SensorNode(sensor=sensor_set[3])

    node_F_tracker = copy.deepcopy(tracker)
    node_F_tracker.detector = FusionQueue()
    node_F = FusionNode(tracker=node_F_tracker, fusion_queue=node_F_tracker.detector, latency=0)

    node_H = SensorNode(sensor=sensor_set[4])

    node_G = FusionNode(tracker=track_tracker, fusion_queue=fusion_queue, latency=0)

    return {'a': node_A, 'b': node_B, 'c': node_C, 'd': node_D, 'e': node_E, 'f': node_F,
            'g': node_G, 'h': node_H}


@pytest.fixture
def edge_lists(nodes, radar_nodes):
    hierarchical_edges = Edges([Edge((nodes['s2'], nodes['s1'])), Edge((nodes['s3'], nodes['s1'])),
                                Edge((nodes['s4'], nodes['s2'])), Edge((nodes['s5'], nodes['s2'])),
                                Edge((nodes['s6'], nodes['s3'])), Edge((nodes['s7'], nodes['s6']))])

    centralised_edges = Edges(
        [Edge((nodes['s2'], nodes['s1'])), Edge((nodes['s3'], nodes['s1'])),
         Edge((nodes['s4'], nodes['s2'])), Edge((nodes['s5'], nodes['s2'])),
         Edge((nodes['s6'], nodes['s3'])), Edge((nodes['s7'], nodes['s6'])),
         Edge((nodes['s7'], nodes['s5'])), Edge((nodes['s5'], nodes['s3']))])

    simple_edges = Edges([Edge((nodes['s2'], nodes['s1'])), Edge((nodes['s3'], nodes['s1']))])

    linear_edges = Edges([Edge((nodes['s1'], nodes['s2'])), Edge((nodes['s2'], nodes['s3'])),
                          Edge((nodes['s3'], nodes['s4'])),
                          Edge((nodes['s4'], nodes['s5']))])

    decentralised_edges = Edges(
        [Edge((nodes['s2'], nodes['s1'])), Edge((nodes['s3'], nodes['s1'])),
         Edge((nodes['s3'], nodes['s4'])), Edge((nodes['s3'], nodes['s5'])),
         Edge((nodes['s5'], nodes['s4']))])

    disconnected_edges = Edges([Edge((nodes['s2'], nodes['s1'])), Edge((nodes['s4'], nodes['s3']))])

    k4_edges = Edges(
        [Edge((nodes['s1'], nodes['s2'])), Edge((nodes['s1'], nodes['s3'])),
         Edge((nodes['s1'], nodes['s4'])), Edge((nodes['s2'], nodes['s3'])),
         Edge((nodes['s2'], nodes['s4'])), Edge((nodes['s3'], nodes['s4']))])

    circular_edges = Edges(
        [Edge((nodes['s1'], nodes['s2'])), Edge((nodes['s2'], nodes['s3'])),
         Edge((nodes['s3'], nodes['s4'])), Edge((nodes['s4'], nodes['s5'])),
         Edge((nodes['s5'], nodes['s1']))])

    disconnected_loop_edges = Edges(
        [Edge((nodes['s2'], nodes['s1'])), Edge((nodes['s4'], nodes['s3'])),
         Edge((nodes['s3'], nodes['s4']))])

    radar_edges = Edges([Edge((radar_nodes['a'], radar_nodes['c'])),
                         Edge((radar_nodes['b'], radar_nodes['c'])),
                         Edge((radar_nodes['d'], radar_nodes['f'])),
                         Edge((radar_nodes['e'], radar_nodes['f'])),
                         Edge((radar_nodes['c'], radar_nodes['g']), edge_latency=0),
                         Edge((radar_nodes['f'], radar_nodes['g']), edge_latency=0),
                         Edge((radar_nodes['h'], radar_nodes['g']))])

    return {"hierarchical_edges": hierarchical_edges, "centralised_edges": centralised_edges,
            "simple_edges": simple_edges, "linear_edges": linear_edges,
            "decentralised_edges": decentralised_edges, "disconnected_edges": disconnected_edges,
            "k4_edges": k4_edges, "circular_edges": circular_edges,
            "disconnected_loop_edges": disconnected_loop_edges, "radar_edges": radar_edges}


