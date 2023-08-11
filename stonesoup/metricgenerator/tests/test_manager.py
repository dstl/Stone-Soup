import datetime
import numpy as np

from ..manager import MultiManager, SimpleManager
from ..base import MetricGenerator
from ...dataassociator import Associator
from ...types.association import Association
from ...types.detection import Detection
from ...types.groundtruth import GroundTruthPath
from ...types.metric import Metric
from ...types.state import State
from ...types.track import Track
from ...base import Property


class DummyMetricGenerator(MetricGenerator):
    generator_name: str = Property(default='test_generator')
    tracks_key: str = Property(default='tracks')
    truths_key: str = Property(default='groundtruth_paths')

    def compute_metric(self, manager, *args, **kwargs):
        return Metric(title="Test metric1",
                      value=25,
                      generator=self)


class DummyAssociator(Associator):

    @staticmethod
    def associate_tracks(tracks, truths):

        associations = set()
        for track in tracks:
            for truth in truths:
                associations.add(Association({track, truth}))
        return associations


def test_add_data_multimanager():
    manager = MultiManager([])

    # Check adding data to empty manager
    tracks = [Track(
        states=State(np.array([[1]]), timestamp=datetime.datetime.now()))]
    truths = [GroundTruthPath(
        states=State(np.array([[2]]), timestamp=datetime.datetime.now()))]
    dets = [Detection(np.array([[3]]), timestamp=datetime.datetime.now())]

    manager.add_data({'truths': truths, 'tracks': tracks, 'detections': dets})

    assert manager.states_sets['tracks'] == set(tracks)
    assert manager.states_sets['truths'] == set(truths)
    assert manager.states_sets['detections'] == set(dets)

    # Check not overwriting data (flag is true by default)
    tracks2 = [Track(
        states=State(np.array([[21]]), timestamp=datetime.datetime.now()))]
    truths2 = [GroundTruthPath(
        states=State(np.array([[22]]), timestamp=datetime.datetime.now()))]
    dets2 = [Detection(np.array([[23]]), timestamp=datetime.datetime.now())]

    manager.add_data({'truths': truths2, 'tracks': tracks2, 'detections': dets2}, overwrite=False)

    assert manager.states_sets['tracks'] == set(tracks + tracks2)
    assert manager.states_sets['truths'] == set(truths + truths2)
    assert manager.states_sets['detections'] == set(dets + dets2)

    # Check overwriting data correctly and check multiple tracks added
    manager.add_data({'truths': truths2,
                      'tracks': tracks,
                      'tracks2': tracks2,
                      'detections': dets2}, overwrite=True)

    assert manager.states_sets['tracks'] == set(tracks)
    assert manager.states_sets['tracks2'] == set(tracks2)
    assert manager.states_sets['truths'] == set(truths2)
    assert manager.states_sets['detections'] == set(dets2)

    # Check adding additional data including repeated data
    manager.add_data({'truths': truths, 'tracks': tracks, 'detections': dets})
    manager.add_data({'truths': truths + truths2, 'tracks': tracks + tracks2,
                      'detections': dets + dets2}, overwrite=False)

    assert manager.states_sets['tracks'] == set(tracks2 + tracks)
    assert manager.states_sets['truths'] == set(truths2 + truths)
    assert manager.states_sets['detections'] == set(dets2 + dets)


def test_add_data_simplemanager():
    manager = SimpleManager([])

    # Check adding data to empty manager
    tracks = [Track(
        states=State(np.array([[1]]), timestamp=datetime.datetime.now()))]
    truths = [GroundTruthPath(
        states=State(np.array([[2]]), timestamp=datetime.datetime.now()))]
    dets = [Detection(np.array([[3]]), timestamp=datetime.datetime.now())]

    manager.add_data(truths, tracks, dets)

    assert manager.states_sets['tracks'] == set(tracks)
    assert manager.states_sets['groundtruth_paths'] == set(truths)
    assert manager.states_sets['detections'] == set(dets)

    # Check not overwriting data (flag is true by default)

    tracks2 = [Track(
        states=State(np.array([[21]]), timestamp=datetime.datetime.now()))]
    truths2 = [GroundTruthPath(
        states=State(np.array([[22]]), timestamp=datetime.datetime.now()))]
    dets2 = [Detection(np.array([[23]]), timestamp=datetime.datetime.now())]

    manager.add_data(truths2, tracks2, dets2, overwrite=False)

    assert manager.states_sets['tracks'] == set(tracks + tracks2)
    assert manager.states_sets['groundtruth_paths'] == set(truths + truths2)
    assert manager.states_sets['detections'] == set(dets + dets2)

    # Check adding additional data including repeated data
    manager = SimpleManager([])
    manager.add_data(truths, tracks, dets)
    manager.add_data(truths + truths2, tracks + tracks2, dets + dets2, overwrite=False)

    assert manager.states_sets['tracks'] == set(tracks2 + tracks)
    assert manager.states_sets['groundtruth_paths'] == set(truths2 + truths)
    assert manager.states_sets['detections'] == set(dets2 + dets)


def test_associate_tracks_multimanager():
    generator = DummyMetricGenerator(generator_name='test_generator',
                                     tracks_key='tracks',
                                     truths_key='truths'
                                     )

    manager = MultiManager(associator=DummyAssociator(), generators=[generator])
    tracks = {Track(
        states=State(np.array([[1]]), timestamp=datetime.datetime.now()))}
    truths = {GroundTruthPath(
        states=State(np.array([[2]]), timestamp=datetime.datetime.now()))}
    manager.add_data({'truths': truths, 'tracks': tracks})

    manager.associate_tracks(generator)

    assert manager.association_set.pop().objects == tracks | truths


def test_associate_tracks_simplemanager():
    generator = DummyMetricGenerator()
    manager = SimpleManager(associator=DummyAssociator(), generators=[generator])
    tracks = {Track(
        states=State(np.array([[1]]), timestamp=datetime.datetime.now()))}
    truths = {GroundTruthPath(
        states=State(np.array([[2]]), timestamp=datetime.datetime.now()))}
    manager.add_data(truths, tracks)

    manager.associate_tracks(generator)

    assert manager.association_set.pop().objects == tracks | truths


def test_list_timestamps_multimanager():

    generator = DummyMetricGenerator(generator_name='test_generator',
                                     tracks_key='tracks',
                                     truths_key='truths'
                                     )

    timestamp1 = datetime.datetime.now()
    timestamp2 = timestamp1 + datetime.timedelta(seconds=10)
    manager = MultiManager(generators=[generator])
    tracks = [Track(
        states=[State(np.array([[1]]), timestamp=timestamp1)])]
    truths = [GroundTruthPath(
        states=[State(np.array([[2]]), timestamp=timestamp2)])]
    manager.add_data({'truths': truths, 'tracks': tracks})

    assert manager.list_timestamps(generator) == [timestamp1, timestamp2]


def test_list_timestamps_simplemanager():

    generator = DummyMetricGenerator()

    timestamp1 = datetime.datetime.now()
    timestamp2 = timestamp1 + datetime.timedelta(seconds=10)
    manager = SimpleManager(generators=[generator])
    tracks = [Track(
        states=[State(np.array([[1]]), timestamp=timestamp1)])]
    truths = [GroundTruthPath(
        states=[State(np.array([[2]]), timestamp=timestamp2)])]
    manager.add_data(truths, tracks)

    assert manager.list_timestamps(generator) == [timestamp1, timestamp2]


def test_generate_metrics_multimanager():
    class DummyGenerator1(MetricGenerator):
        generator_name: str = Property()

        def compute_metric(self, *args, **kwargs):
            return Metric(title="Test metric2",
                          value=25,
                          generator=self)

    class DummyGenerator2(MetricGenerator):
        generator_name: str = Property()

        def compute_metric(self, *args, **kwargs):
            return Metric(title="Test metric3 at times",
                          value=50,
                          generator=self)

    generator1 = DummyGenerator1(generator_name='generator1')
    generator2 = DummyGenerator2(generator_name='generator2')
    manager = MultiManager(generators=[generator1, generator2])
    metrics = manager.generate_metrics()

    # test metrics returned correctly
    assert isinstance(metrics, dict)
    assert len(metrics) == 2
    assert list(metrics.keys()) == ['generator1', 'generator2']

    metric2 = metrics['generator1']
    metric3 = metrics['generator2']

    # test metric content generated correctly from generator1
    assert isinstance(metric2['Test metric2'], Metric)
    assert list(metric2.keys()) == ["Test metric2"]
    assert metrics['generator1'].get("Test metric2") == metric2['Test metric2']
    assert np.array_equal(metric2['Test metric2'].value, 25)
    assert metric2['Test metric2'].generator == generator1

    # test metric content generated correctly from generator2
    assert isinstance(metric3['Test metric3 at times'], Metric)
    assert list(metric3.keys()) == ["Test metric3 at times"]
    assert metrics['generator2'].get("Test metric3 at times") == metric3['Test metric3 at times']
    assert np.array_equal(metric3['Test metric3 at times'].value, 50)
    assert metric3['Test metric3 at times'].generator == generator2


def test_generate_metrics_simplemanager():
    class DummyGenerator1(MetricGenerator):
        generator_name: str = Property()

        def compute_metric(self, *args, **kwargs):
            return Metric(title="Test metric4",
                          value=30,
                          generator=self)

    class DummyGenerator2(MetricGenerator):
        generator_name: str = Property()

        def compute_metric(self, *args, **kwargs):
            return Metric(title="Test metric5 at times",
                          value=60,
                          generator=self)

    generator1 = DummyGenerator1(generator_name='generator1')
    generator2 = DummyGenerator2(generator_name='generator2')
    manager = SimpleManager(generators=[generator1, generator2])
    metrics = manager.generate_metrics()

    # test metrics has returned correctly
    assert isinstance(metrics, dict)
    assert len(metrics) == 2
    assert list(metrics.keys()) == ['Test metric4', 'Test metric5 at times']

    metric4 = metrics['Test metric4']
    metric5 = metrics['Test metric5 at times']

    # test metric content generated correctly from generator1
    assert isinstance(metric4, Metric)
    assert metric4.title == 'Test metric4'
    assert metrics.get("Test metric4") == metric4
    assert np.array_equal(metric4.value, 30)
    assert metric4.generator == generator1

    # test metric content generated correctly from generator2
    assert isinstance(metric5, Metric)
    assert metric5.title == 'Test metric5 at times'
    assert metrics.get("Test metric5 at times") == metric5
    assert np.array_equal(metric5.value, 60)
    assert metric5.generator == generator2
