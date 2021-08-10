import pytest

from ..opensky import OpenSkyNetworkDetectionReader, OpenSkyNetworkGroundTruthReader

pytestmark = pytest.mark.remote_data


@pytest.mark.parametrize(
    'reader_type',
    (OpenSkyNetworkDetectionReader, OpenSkyNetworkGroundTruthReader))
@pytest.mark.parametrize(
    'bbox',
    [None, (-7.57216793459, 49.959999905, 1.68153079591, 58.6350001085)],
    ids=['None', 'GB'])
def test_opensky_reader(reader_type, bbox):
    reader = reader_type(bbox)

    prev_time = None
    for n, (time, states) in enumerate(reader, 1):
        if prev_time is not None:
            assert time > prev_time
        prev_time = time

        for state in states:
            if bbox:
                assert bbox[0] < state.state_vector[0] < bbox[2]
                assert bbox[1] < state.state_vector[1] < bbox[3]

            # When using GroundTruthReader, and ID looks like ICAO24 (ignore those missing ICAO24)
            if isinstance(reader_type, OpenSkyNetworkGroundTruthReader) and len(state.id) == 6:
                assert all(sub_state.metadata['icao24'] == state.id for sub_state in state)

        if isinstance(reader_type, OpenSkyNetworkGroundTruthReader):
            assert any(len(path) == n for path in states)
            assert all(len(path) <= n for path in states)

        if n > 3:
            break
