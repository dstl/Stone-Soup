import datetime
import numpy as np

from ..manager import MultiManager
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
    generator_name: str = Property()
    tracks_key: str = Property()
    truths_key: str = Property()

    def compute_metric(self, manager, *args, **kwargs):
        return Metric(title="Test metric1",
                      value=25,
                      generator=self)

def test_add_data():
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

    # Check adding additional data including repeated data
    manager = MultiManager([])
    manager.add_data({'truths': truths, 'tracks': tracks, 'detections': dets})
    manager.add_data({'truths': truths + truths2, 'tracks': tracks + tracks2, 'detections': dets + dets2},
                     overwrite=True)

    assert manager.states_sets['tracks'] == set(tracks2 + tracks)
    assert manager.states_sets['truths'] == set(truths2 + truths)
    assert manager.states_sets['detections'] == set(dets2 + dets)


def test_associate_tracks():
    class DummyAssociator(Associator):

        def associate_tracks(self, tracks, truths):

            associations = set()
            for track in tracks:
                for truth in truths:
                    associations.add(Association({track, truth}))
            return associations

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


def test_list_timestamps():

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


def test_generate_metrics():
    class DummyGenerator1(MetricGenerator):
        generator_name: str = Property()
        tracks_key: str = Property()
        truths_key: str = Property()

        def compute_metric(self, manager, *args, **kwargs):
            return Metric(title="Test metric2",
                          value=25,
                          generator=self)

    class DummyGenerator2(MetricGenerator):
        generator_name: str = Property()
        tracks_key: str = Property()
        truths_key: str = Property()

        def compute_metric(self, manager, *args, **kwargs):
            return Metric(title="Test metric3 at times",
                          value=50,
                          generator=self)

    generator1 = DummyGenerator1(generator_name='generator1',
                                 tracks_key='tracks',
                                 truths_key='truths'
                                 )
    generator2 = DummyGenerator2(generator_name='generator2',
                                 tracks_key='tracks',
                                 truths_key='truths'
                                 )

    manager = MultiManager(generators=[generator1, generator2])

    metrics = manager.generate_metrics()
    metric1 = metrics['generator1']
    metric2 = metrics['generator2']
    # metric1 = [metrics.get(i) for i in metrics if metrics[i].generator == generator1][0]
    # metric2 = [metrics.get(i) for i in metrics if metrics[i].generator == generator2][0]

    assert len(metrics) == 2
    assert list(metric1.keys())[0] == "Test metric2"
    assert metrics['generator1'].get("Test metric2") == metric1['Test metric2']
    assert np.array_equal(metric1['Test metric2'].value, 25)
    assert np.array_equal(metric1['Test metric2'].generator, generator1)
    assert list(metric2.keys())[0] == "Test metric3 at times"
    assert metrics['generator2'].get("Test metric3 at times") == metric2['Test metric3 at times']
    assert np.array_equal(metric2['Test metric3 at times'].value, 50)
    assert np.array_equal(metric2['Test metric3 at times'].generator, generator2)
