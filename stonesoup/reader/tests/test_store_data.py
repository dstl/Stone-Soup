import numpy as np

from ..generic import DictionaryGroundTruthReader, DictionaryTrackReader


def _groundtruth_reader(store_data=False):
    return DictionaryGroundTruthReader(
        dictionaries=[
            {'id': 'A', 'x': 0, 'y': 1, 'time': '2026-01-01T00:00:00'},
            {'id': 'A', 'x': 1, 'y': 2, 'time': '2026-01-01T00:01:00'},
            {'id': 'B', 'x': 3, 'y': 4, 'time': '2026-01-01T00:01:00'},
        ],
        state_vector_fields=['x', 'y'],
        time_field='time',
        path_id_field='id',
        store_data=store_data,
    )


def _track_reader(store_data=False):
    return DictionaryTrackReader(
        dictionaries=[
            {'id': 'A', 'x': 0, 'y': 1, 'time': '2026-01-01T00:00:00'},
            {'id': 'A', 'x': 1, 'y': 2, 'time': '2026-01-01T00:01:00'},
            {'id': 'B', 'x': 3, 'y': 4, 'time': '2026-01-01T00:01:00'},
        ],
        state_vector_fields=['x', 'y'],
        time_field='time',
        track_id_field='id',
        default_covar=np.eye(2),
        covar_fields_index={},
        store_data=store_data,
    )


def test_groundtruth_reader_store_data():
    reader = _groundtruth_reader(store_data=True)
    list(reader)

    assert set(reader.groundtruth_dict) == {'A', 'B'}
    assert len(reader.groundtruth_dict['A']) == 2
    assert len(reader.groundtruth_dict['B']) == 1


def test_groundtruth_reader_does_not_store_data_by_default():
    reader = _groundtruth_reader()
    list(reader)

    assert reader.groundtruth_dict is None


def test_track_reader_store_data():
    reader = _track_reader(store_data=True)
    list(reader)

    assert set(reader.track_dict) == {'A', 'B'}
    assert len(reader.track_dict['A']) == 2
    assert len(reader.track_dict['B']) == 1


def test_track_reader_does_not_store_data_by_default():
    reader = _track_reader()
    list(reader)

    assert reader.track_dict is None
