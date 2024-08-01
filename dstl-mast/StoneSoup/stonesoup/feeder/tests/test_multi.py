import pytest

from ..multi import MultiDataFeeder


@pytest.mark.parametrize('n', [1, 2, 3])
def test_multi_detections(n, reader):
    single_time_list = list()
    multi_time_list = list()

    multi_detector = MultiDataFeeder([reader]*n)
    for single_iterations, (time, _) in enumerate(reader, 1):
        single_time_list.append(time)
    for multi_iterations, (time, _) in enumerate(multi_detector, 1):
        multi_time_list.append(time)
    assert multi_iterations == single_iterations * n
    # Compare every other element but skip last ones due to out of sequence
    # measurements coming from detector.
    assert multi_time_list[:-n:n] == single_time_list[:-1]
    assert multi_time_list[-1] == single_time_list[-1]
