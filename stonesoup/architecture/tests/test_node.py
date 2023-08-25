import pytest

import copy

from ..node import Node, SensorNode, FusionNode, SensorFusionNode, RepeaterNode
from ..edge import FusionQueue
from ...types.hypothesis import Hypothesis
from ...types.track import Track
from ...types.state import State, StateVector


def test_node(data_pieces, times, nodes):
    node = Node()
    assert node.latency == 0.0
    assert node.font_size == 5
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


def test_sensor_node(nodes):
    with pytest.raises(TypeError):
        SensorNode()

    sensor = nodes['s1'].sensor
    snode = SensorNode(sensor=sensor)
    assert snode.sensor == sensor
    assert snode.colour == '#1f77b4'
    assert snode.shape == 'oval'
    assert snode.node_dim == (0.5, 0.3)


def test_fusion_node(tracker):
    fnode_tracker = copy.deepcopy(tracker)
    fnode_tracker.detector = FusionQueue()
    fnode = FusionNode(tracker=fnode_tracker, fusion_queue=fnode_tracker.detector, latency=0)

    assert fnode.colour == '#006400'
    assert fnode.shape == 'hexagon'
    assert fnode.node_dim == (0.6, 0.3)
    assert fnode.tracks == set()

    with pytest.raises(TypeError):
        FusionNode()

    fnode.fuse()  # Works. Thorough testing left to test_architecture.py


def test_sf_node(tracker, nodes):
    with pytest.raises(TypeError):
        SensorFusionNode()
    sfnode_tracker = copy.deepcopy(tracker)
    sfnode_tracker.detector = FusionQueue()
    sfnode = SensorFusionNode(tracker=sfnode_tracker, fusion_queue=sfnode_tracker.detector,
                              latency=0, sensor=nodes['s1'].sensor)

    assert sfnode.colour == '#909090'
    assert sfnode.shape == 'rectangle'
    assert sfnode.node_dim == (0.1, 0.3)

    assert sfnode.tracks == set()


def test_repeater_node():
    rnode = RepeaterNode()

    assert rnode.colour == '#ff7f0e'
    assert rnode.shape == 'circle'
    assert rnode.node_dim == (0.5, 0.3)