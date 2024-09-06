import numpy as np
import pytest

from stonesoup.architecture.edge import FusionQueue
from stonesoup.architecture.generator import InformationArchitectureGenerator, \
    NetworkArchitectureGenerator
from stonesoup.sensor.sensor import Sensor
from stonesoup.tracker import Tracker


def test_info_arch_gen_init(generator_params):
    start_time = generator_params['start_time']
    base_sensor = generator_params['base_sensor']
    base_tracker = generator_params['base_tracker']

    gen = InformationArchitectureGenerator(start_time=start_time,
                                           mean_degree=2,
                                           node_ratio=[3, 1, 1],
                                           base_tracker=base_tracker,
                                           base_sensor=base_sensor)

    # Test default values
    assert gen.arch_type == 'decentralised'
    assert gen.iteration_limit == 10000
    assert gen.allow_invalid_graph is False
    assert gen.n_archs == 2

    # Test variables set in __init__()
    assert gen.n_sensor_nodes == 3
    assert gen.n_sensor_fusion_nodes == 1
    assert gen.n_fusion_nodes == 1

    assert gen.sensor_max_distance == (0, 0)

    with pytest.raises(ValueError):
        InformationArchitectureGenerator(arch_type='not_valid',
                                         start_time=start_time,
                                         mean_degree=2,
                                         node_ratio=[3, 1, 1],
                                         base_tracker=base_tracker,
                                         base_sensor=base_sensor)


def test_info_generate_hierarchical(generator_params):
    start_time = generator_params['start_time']
    base_sensor = generator_params['base_sensor']
    base_tracker = generator_params['base_tracker']

    gen = InformationArchitectureGenerator(arch_type='hierarchical',
                                           start_time=start_time,
                                           mean_degree=2,
                                           node_ratio=[2, 2, 1],
                                           base_tracker=base_tracker,
                                           base_sensor=base_sensor,
                                           n_archs=3)

    archs = gen.generate()

    for arch in archs:

        # Check correct number of nodes
        assert len(arch.all_nodes) == sum([2, 2, 1])

        # Check node types
        assert len(arch.fusion_nodes) == sum([2, 1])
        assert len(arch.sensor_nodes) == sum([2, 2])

        assert len(arch.edges) == sum([2, 2, 1]) - 1

        for node in arch.fusion_nodes:
            # Check each fusion node has a tracker
            assert isinstance(node.tracker, Tracker)
            # Check each tracker has a FusionQueue
            assert isinstance(node.tracker.detector.reader, FusionQueue)

        for node in arch.sensor_nodes:
            # Check each sensor node has a Sensor
            assert isinstance(node.sensor, Sensor)


def test_info_generate_decentralised(generator_params):
    start_time = generator_params['start_time']
    base_sensor = generator_params['base_sensor']
    base_tracker = generator_params['base_tracker']

    mean_deg = 2.5

    gen = InformationArchitectureGenerator(arch_type='decentralised',
                                           start_time=start_time,
                                           mean_degree=mean_deg,
                                           node_ratio=[3, 1, 1],
                                           base_tracker=base_tracker,
                                           base_sensor=base_sensor,
                                           n_archs=2)

    archs = gen.generate()

    for arch in archs:

        # Check correct number of nodes
        assert len(arch.all_nodes) == sum([3, 1, 1])

        # Check node types
        assert len(arch.fusion_nodes) == sum([1, 1])
        assert len(arch.sensor_nodes) == sum([3, 1])

        assert len(arch.edges) == np.ceil(sum([3, 1, 1]) * mean_deg * 0.5)

        for node in arch.fusion_nodes:
            # Check each fusion node has a tracker
            assert isinstance(node.tracker, Tracker)
            # Check each tracker has a FusionQueue
            assert isinstance(node.tracker.detector.reader, FusionQueue)

        for node in arch.sensor_nodes:
            # Check each sensor node has a Sensor
            assert isinstance(node.sensor, Sensor)


def test_info_generate_invalid(generator_params):
    start_time = generator_params['start_time']
    base_sensor = generator_params['base_sensor']
    base_tracker = generator_params['base_tracker']

    mean_deg = 2.5

    with pytest.raises(ValueError):
            gen = InformationArchitectureGenerator(arch_type='invalid',
                                           start_time=start_time,
                                           mean_degree=mean_deg,
                                           node_ratio=[3, 1, 1],
                                           base_tracker=base_tracker,
                                           base_sensor=base_sensor,
                                           n_archs=2)

def test_net_arch_gen_init(generator_params):
    start_time = generator_params['start_time']
    base_sensor = generator_params['base_sensor']
    base_tracker = generator_params['base_tracker']

    gen = NetworkArchitectureGenerator(start_time=start_time,
                                       mean_degree=2,
                                       node_ratio=[3, 1, 1],
                                       base_tracker=base_tracker,
                                       base_sensor=base_sensor)

    # Test default values
    assert gen.arch_type == 'decentralised'
    assert gen.iteration_limit == 10000
    assert gen.allow_invalid_graph is False
    assert gen.n_archs == 2

    # Test variables set in __init__()
    assert gen.n_sensor_nodes == 3
    assert gen.n_sensor_fusion_nodes == 1
    assert gen.n_fusion_nodes == 1

    assert gen.sensor_max_distance == (0, 0)

    with pytest.raises(ValueError):
        NetworkArchitectureGenerator(arch_type='not_valid',
                                     start_time=start_time,
                                     mean_degree=2,
                                     node_ratio=[3, 1, 1],
                                     base_tracker=base_tracker,
                                     base_sensor=base_sensor)


def test_net_generate_hierarchical(generator_params):
    start_time = generator_params['start_time']
    base_sensor = generator_params['base_sensor']
    base_tracker = generator_params['base_tracker']

    gen = NetworkArchitectureGenerator(arch_type='hierarchical',
                                       start_time=start_time,
                                       mean_degree=2,
                                       node_ratio=[3, 1, 1],
                                       base_tracker=base_tracker,
                                       base_sensor=base_sensor,
                                       n_archs=3)

    archs = gen.generate()

    for arch in archs:

        # Check correct number of nodes
        assert len(arch.all_nodes) == sum([3, 1, 1]) + len(arch.repeater_nodes)

        # Check node types
        assert len(arch.fusion_nodes) == sum([1, 1])
        assert len(arch.sensor_nodes) == sum([3, 1])

        assert len(arch.edges) == 2 * len(arch.repeater_nodes)

        for node in arch.fusion_nodes:
            # Check each fusion node has a tracker
            assert isinstance(node.tracker, Tracker)
            # Check each tracker has a FusionQueue
            assert isinstance(node.tracker.detector.reader, FusionQueue)

        for node in arch.sensor_nodes:
            # Check each sensor node has a Sensor
            assert isinstance(node.sensor, Sensor)


def test_net_generate_decentralised(generator_params):
    start_time = generator_params['start_time']
    base_sensor = generator_params['base_sensor']
    base_tracker = generator_params['base_tracker']

    gen = NetworkArchitectureGenerator(arch_type='decentralised',
                                       start_time=start_time,
                                       mean_degree=2,
                                       node_ratio=[3, 1, 1],
                                       base_tracker=base_tracker,
                                       base_sensor=base_sensor,
                                       n_archs=3)

    archs = gen.generate()

    for arch in archs:

        # Check correct number of nodes
        assert len(arch.all_nodes) == sum([3, 1, 1]) + len(arch.repeater_nodes)

        # Check node types
        assert len(arch.fusion_nodes) == sum([1, 1])
        assert len(arch.sensor_nodes) == sum([3, 1])
        for edge in arch.edges:
            print((edge.nodes[0].label, edge.nodes[1].label))
        print(len(arch.repeater_nodes))
        assert len(arch.edges) == 2 * len(arch.repeater_nodes)

        for node in arch.fusion_nodes:
            # Check each fusion node has a tracker
            assert isinstance(node.tracker, Tracker)
            # Check each tracker has a FusionQueue
            assert isinstance(node.tracker.detector.reader, FusionQueue)

        for node in arch.sensor_nodes:
            # Check each sensor node has a Sensor
            assert isinstance(node.sensor, Sensor)
