import pytest

import copy
from datetime import datetime
import numpy as np

from ..node import Node, SensorNode, FusionNode, SensorFusionNode, RepeaterNode
from ..edge import FusionQueue, DataPiece
from ..generator import NetworkArchitectureGenerator
from ...types.hypothesis import Hypothesis
from ...types.track import Track
from ...types.detection import Detection
from ... types.groundtruth import GroundTruthPath
from ...types.state import State, StateVector


def test_node(data_pieces, times, nodes):
    node = Node()
    assert node.latency == 0.0
    assert node.font_size is None
    assert len(node.data_held) == 3
    assert node.data_held == {"fused": {}, "created": {}, "unfused": {}}

    node.update(times['a'], times['b'], data_pieces['a'], "fused")
    new_data_piece = node.data_held['fused'][times['a']].pop()
    assert new_data_piece.originator == nodes['a']
    assert isinstance(new_data_piece.data, Track) and len(new_data_piece.data) == 0
    assert new_data_piece.time_arrived == times['b']

    with pytest.raises(TypeError):
        node.update(times['b'], times['a'], data_pieces['hyp'], "created")
    node.update(times['b'], times['a'], data_pieces['hyp'], "created",
                track=Track([State(state_vector=StateVector([1]))]))
    new_data_piece2 = node.data_held['created'][times['b']].pop()
    assert new_data_piece2.originator == nodes['b']
    assert isinstance(new_data_piece2.data, Hypothesis)
    assert new_data_piece2.time_arrived == times['a']

    with pytest.raises(TypeError):
        node.update(times['a'],
                    times['b'],
                    data_pieces['fail'],
                    "fused",
                    track=Track([]),
                    use_arrival_time=False)


def test_sensor_node(nodes):
    with pytest.raises(TypeError):
        SensorNode()

    sensor = nodes['s1'].sensor
    snode = SensorNode(sensor=sensor)
    assert snode.sensor == sensor
    assert snode.colour == '#006eff'
    assert snode.shape == 'oval'


def test_fusion_node(tracker):
    fnode_tracker = copy.deepcopy(tracker)
    fnode_tracker.detector = FusionQueue()
    fnode = FusionNode(tracker=fnode_tracker, fusion_queue=fnode_tracker.detector, latency=0)

    assert fnode.colour == '#00b53d'
    assert fnode.shape == 'hexagon'
    assert fnode.tracks == set()

    with pytest.raises(TypeError):
        FusionNode()

    fnode.fuse()  # Works. Thorough testing left to test_architecture.py

    # Test FusionNode instantiation with no fusion_queue
    fnode2_tracker = copy.deepcopy(tracker)
    fnode2_tracker.detector = FusionQueue()
    fnode2 = FusionNode(fnode2_tracker)

    assert fnode2.fusion_queue == fnode2_tracker.detector

    # Test FusionNode instantiation with no fusion_queue or tracker.detector
    fnode3_tracker = copy.deepcopy(tracker)
    fnode3_tracker.detector = None
    fnode3 = FusionNode(fnode3_tracker)

    assert isinstance(fnode3.fusion_queue, FusionQueue)


def test_sf_node(tracker, nodes):
    with pytest.raises(TypeError):
        SensorFusionNode()
    sfnode_tracker = copy.deepcopy(tracker)
    sfnode_tracker.detector = FusionQueue()
    sfnode = SensorFusionNode(tracker=sfnode_tracker, fusion_queue=sfnode_tracker.detector,
                              latency=0, sensor=nodes['s1'].sensor)

    assert sfnode.colour == '#fc9000'
    assert sfnode.shape == 'diamond'

    assert sfnode.tracks == set()


def test_repeater_node():
    rnode = RepeaterNode()

    assert rnode.colour == '#909090'
    assert rnode.shape == 'rectangle'


def test_update(tracker):
    A = Node()
    B = Node()
    C = Node()

    dt0 = "This ain't no datetime object"
    dt1 = datetime.now()

    t_data = DataPiece(A, A, Track([]), dt1)
    d_data = DataPiece(A, A, Detection(state_vector=StateVector(np.random.rand(4, 1)),
                                       timestamp=dt1), dt1)
    h_data = DataPiece(A, A, Hypothesis(), dt1)

    # Test invalid time inputs
    with pytest.raises(TypeError):
        A.update(dt0, dt0, 'faux DataPiece', 'created')

    # Test invalid data_piece
    with pytest.raises(TypeError):
        A.update(dt1, dt1, 'faux DataPiece', 'created')

    # Test invalid category
    with pytest.raises(ValueError):
        A.update(dt1, dt1, t_data, 'forged')

    # Test non-detection-or-track-datapiece with Track=False
    with pytest.raises(TypeError):
        A.update(dt1, dt1, h_data, 'created')

    # Test non-hypothesis-datapiece with Track=True
    with pytest.raises(TypeError):
        A.update(dt1, dt1, d_data, 'created', track=True)

    # For track DataPiece, test new DataPiece is created and placed in data_held
    A.update(dt1, dt1, t_data, 'created')
    new_data_piece = A.data_held['created'][dt1].pop()

    assert t_data.originator == new_data_piece.originator
    assert t_data.data == new_data_piece.data
    assert t_data.time_arrived == new_data_piece.time_arrived

    # For detection DataPiece, test new DataPiece is created and placed in data_held
    B.update(dt1, dt1, d_data, 'created')
    new_data_piece = B.data_held['created'][dt1].pop()

    assert d_data.originator == new_data_piece.originator
    assert d_data.data == new_data_piece.data
    assert d_data.time_arrived == new_data_piece.time_arrived

    # For hypothesis DataPiece, test new DataPiece is created and placed in data_held
    C.update(dt1, dt1, h_data, 'created', track=True)
    new_data_piece = C.data_held['created'][dt1].pop()

    assert h_data.originator == new_data_piece.originator
    assert h_data.data == new_data_piece.data
    assert h_data.time_arrived == new_data_piece.time_arrived

    # Test placing data into fusion queue - use_arrival_time=False
    D = FusionNode(tracker=tracker)
    D.update(dt1, dt1, d_data, 'created', use_arrival_time=False)
    assert d_data.data in D.fusion_queue.received

    # Test placing data into fusion queue - use_arrival_time=True
    D = FusionNode(tracker=tracker)
    D.update(dt1, dt1, d_data, 'created', use_arrival_time=True)
    copied_data = D.fusion_queue.received.pop()
    assert sum(copied_data.state_vector - d_data.data.state_vector) == 0
    assert copied_data.measurement_model == d_data.data.measurement_model
    assert copied_data.metadata == d_data.data.metadata


def test_fuse(generator_params, ground_truths, timesteps):
    # Full data fusion simulation
    start_time = generator_params['start_time']
    base_sensor = generator_params['base_sensor']
    base_tracker = generator_params['base_tracker']
    gen = NetworkArchitectureGenerator(arch_type='hierarchical',
                                       start_time=start_time,
                                       mean_degree=2,
                                       node_ratio=[3, 2, 1],
                                       base_tracker=base_tracker,
                                       base_sensor=base_sensor,
                                       n_archs=1,
                                       sensor_max_distance=(10, 10))

    arch = gen.generate()[0]

    assert all([isinstance(gt, GroundTruthPath) for gt in ground_truths])

    for time in timesteps:
        arch.measure(ground_truths, noise=True)
        arch.propagate(time_increment=1)

    for node in arch.fusion_nodes:
        for track in node.tracks:
            assert isinstance(track, Track)
