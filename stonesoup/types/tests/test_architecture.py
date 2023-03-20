from ..architecture import Architecture, NetworkArchitecture, InformationArchitecture, \
    CombinedArchitecture, FusionNode, RepeaterNode, SensorNode, Node, SensorFusionNode, Edge, Edges

from ...sensor.base import PlatformMountable
from stonesoup.models.measurement.categorical import MarkovianMeasurementModel
from stonesoup.models.transition.categorical import MarkovianTransitionModel
from stonesoup.types.groundtruth import CategoricalGroundTruthState
from stonesoup.types.state import CategoricalState
from stonesoup.types.groundtruth import GroundTruthPath
from stonesoup.sensor.categorical import HMMSensor
from stonesoup.predictor.categorical import HMMPredictor
from stonesoup.updater.categorical import HMMUpdater
from stonesoup.hypothesiser.categorical import HMMHypothesiser
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.initiator.categorical import SimpleCategoricalMeasurementInitiator
from stonesoup.deleter.time import UpdateTimeStepsDeleter

from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

import pytest

@pytest.fixture
def params():
    e = np.array([[0.8, 0.1],  # P(small | bike), P(small | car)
                  [0.19, 0.3],  # P(medium | bike), P(medium | car)
                  [0.01, 0.6]])  # P(large | bike), P(large | car)
    model = MarkovianMeasurementModel(emission_matrix=e,
                                      measurement_categories=['small', 'medium', 'large'])
    hmm_sensor = HMMSensor(measurement_model=model)
    nodes = SensorNode(sensor=hmm_sensor), SensorNode(sensor=hmm_sensor), SensorNode(hmm_sensor), \
        SensorNode(hmm_sensor)

    return {"small_nodes": nodes} # big nodes cna be added


def test_info_architecure_propagation(params):
    """Is information correctly propagated through the architecture?"""
    nodes = params['small_nodes']
    # A "Y" shape, with data travelling "up" the Y
    # First, with no latency
    edges = Edges([Edge((nodes[0], nodes[1])), Edge((nodes[1], nodes[2])),
                   Edge((nodes[1], nodes[3]))])
    architecture = InformationArchitecture(edges=edges)


def test_info_architecture_fusion(params):
    """Are Fusion/SensorFusionNodes correctly fusing information together.
    (Currently they won't be)"""


def test_information_architecture_using_hmm():
    """Heavily inspired by the example: "Classifying Using HMM"""

    # Skip to line 89 for network architectures (rest is from hmm example)

    transition_matrix = np.array([[0.8, 0.2],  # P(bike | bike), P(bike | car)
                                  [0.4, 0.6]])  # P(car | bike), P(car | car)
    category_transition = MarkovianTransitionModel(transition_matrix=transition_matrix)

    start = datetime.now()

    hidden_classes = ['bike', 'car']

    # Generating ground truth
    ground_truths = list()
    for i in range(1, 4):  # 4 targets
        state_vector = np.zeros(2)  # create a vector with 2 zeroes
        state_vector[
            np.random.choice(2, 1, p=[1 / 2, 1 / 2])] = 1  # pick a random class out of the 2
        ground_truth_state = CategoricalGroundTruthState(state_vector,
                                                         timestamp=start,
                                                         categories=hidden_classes)

        ground_truth = GroundTruthPath([ground_truth_state], id=f"GT{i}")

        for _ in range(10):
            new_vector = category_transition.function(ground_truth[-1],
                                                      noise=True,
                                                      time_interval=timedelta(seconds=1))
            new_state = CategoricalGroundTruthState(
                new_vector,
                timestamp=ground_truth[-1].timestamp + timedelta(seconds=1),
                categories=hidden_classes
            )

            ground_truth.append(new_state)
        ground_truths.append(ground_truth)

    E = np.array([[0.8, 0.1],  # P(small | bike), P(small | car)
                  [0.19, 0.3],  # P(medium | bike), P(medium | car)
                  [0.01, 0.6]])  # P(large | bike), P(large | car)
    model = MarkovianMeasurementModel(emission_matrix=E,
                                      measurement_categories=['small', 'medium', 'large'])

    hmm_sensor = HMMSensor(measurement_model=model)

    transition_matrix = np.array([[0.81, 0.19],  # P(bike | bike), P(bike | car)
                                  [0.39, 0.61]])  # P(car | bike), P(car | car)
    category_transition = MarkovianTransitionModel(transition_matrix=transition_matrix)

    predictor = HMMPredictor(category_transition)

    updater = HMMUpdater()

    hypothesiser = HMMHypothesiser(predictor=predictor, updater=updater)

    data_associator = GNNWith2DAssignment(hypothesiser)

    prior = CategoricalState([1 / 2, 1 / 2], categories=hidden_classes)

    initiator = SimpleCategoricalMeasurementInitiator(prior_state=prior, updater=updater)

    deleter = UpdateTimeStepsDeleter(2)

    # START HERE FOR THE GOOD STUFF

    hmm_sensor_node_A = SensorNode(sensor=hmm_sensor)
    hmm_sensor_processing_node_B = SensorProcessingNode(sensor=hmm_sensor, predictor=predictor,
                                                        updater=updater, hypothesiser=hypothesiser,
                                                        data_associator=data_associator,
                                                        initiator=initiator, deleter=deleter)
    info_architecture = InformationArchitecture(
        edge_list=[(hmm_sensor_node_A, hmm_sensor_processing_node_B)],
        current_time=start)

    for _ in range(10):
        # Lots going on inside these two
        # Ctrl + click to jump to source code for a class or function :)

        # Gets all SensorNodes (as SensorProcessingNodes inherit from SensorNodes, this is
        # both the Nodes in this example) to measure
        info_architecture.measure(ground_truths, noise=True)
        # The data is propagated through the network, ie our SensorNode sends its measurements to
        # the SensorProcessingNode.
        info_architecture.propagate(time_increment=1)

    # OK, so this runs up to here, but something has gone wrong
    tracks = hmm_sensor_processing_node_B.tracks
    print(len(tracks))
    print(hmm_sensor_processing_node_B.data_held)

    # There is data, but no tracks...

    def plot(path, style):
        times = list()
        probs = list()
        for state in path:
            times.append(state.timestamp)
            probs.append(state.state_vector[0])
        plt.plot(times, probs, linestyle=style)

    # Node B is the 'parent' node, so we want its tracks. Also the only ProcessingNode
    # in this example

    for truth in ground_truths:
        plot(truth, '--')
    for track in tracks:
        plot(track, '-')

    plt.show; #



def test_architecture():
    a, b, c, d, e = RepeaterNode(), RepeaterNode(), RepeaterNode(), RepeaterNode(), \
                    RepeaterNode()

    edge_list_unconnected = [(a, b), (c, d)]
    with pytest.raises(ValueError):
        Architecture(edge_list=edge_list_unconnected, force_connected=True)
    edge_list_connected = [(a, b), (b, c), (b, d)]
    a_test_hier = Architecture(edge_list=edge_list_connected,
                               force_connected=False, name="bleh")
    edge_list_loop = [(a, b), (b, c), (c, a)]
    a_test_loop = Architecture(edge_list=edge_list_loop,
                               force_connected=False)

    assert a_test_loop.is_connected and a_test_hier.is_connected
    assert a_test_hier.is_hierarchical
    assert not a_test_loop.is_hierarchical

    with pytest.raises(TypeError):
        a_test_hier.plot(dir_path='U:\\My Documents\\temp', plot_title=True, use_positions=True)

    a_pos, b_pos, c_pos = RepeaterNode(label="Alpha", position=(1, 2)), \
                          SensorNode(sensor=PlatformMountable(), position=(1, 1)), \
                          RepeaterNode(position=(2, 1))
    edge_list_pos = [(a_pos, b_pos), (b_pos, c_pos), (c_pos, a_pos)]
    pos_test = NetworkArchitecture(edge_list_pos)
    pos_test.plot(dir_path='C:\\Users\\orosoman\\Desktop\\arch_plots', plot_title=True,
                  use_positions=True)


def test_information_architecture():
    with pytest.raises(TypeError):
        # Repeater nodes have no place in an information architecture
        InformationArchitecture(edge_list=[(RepeaterNode(), RepeaterNode())])
    ia_test = InformationArchitecture()


def test_density():
    a, b, c, d = RepeaterNode(), RepeaterNode(), RepeaterNode(), RepeaterNode()
    edge_list = [(a, b), (c, d), (d, a)]
    assert Architecture(edge_list=edge_list, node_set={a, b, c, d}).density == 1/2


