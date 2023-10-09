import datetime

import pytest

from .. import non_state_measures, measures
from ..types.state import State, StateMutableSequence


def test_multiple_measure_raise_error():
    with pytest.raises(TypeError):
        non_state_measures.MultipleMeasure()


def test_track_measure_raise_error():
    with pytest.raises(TypeError):
        non_state_measures.TrackMeasure()


@pytest.fixture
def state_sequence_measure():
    return non_state_measures.StateSequenceMeasure(state_measure=measures.Euclidean())


@pytest.fixture
def sms_1():
    return StateMutableSequence(
        states=[State(state_vector=[idx], timestamp=datetime.datetime(2023, 10, 1, 0, 0, idx))
                for idx in range(10)])


@pytest.fixture
def sms_2():
    return StateMutableSequence(
        states=[State(state_vector=[idx + 1], timestamp=datetime.datetime(2023, 10, 1, 0, 0, idx))
                for idx in range(2, 5)])


@pytest.fixture
def sms_3():
    return StateMutableSequence(
        states=[State(state_vector=[idx * 2], timestamp=datetime.datetime(2023, 10, 1, 0, 0, idx))
                for idx in range(5, 9)])


def test_state_sequence_measure_1(state_sequence_measure, sms_1, sms_2):
    assert state_sequence_measure(sms_1, sms_2) == [1, 1, 1]


def test_state_sequence_measure_2(state_sequence_measure, sms_1, sms_3):
    assert state_sequence_measure(sms_1, sms_3) == [5, 6, 7, 8]


def test_state_sequence_measure_3(state_sequence_measure, sms_1, sms_3):
    times_to_measure = [datetime.datetime(2023, 10, 1, 0, 0, 6),
                        datetime.datetime(2023, 10, 1, 0, 0, 7)]
    assert state_sequence_measure(sms_1, sms_3, times_to_measure) == [6, 7]


def test_state_sequence_measure_warning(state_sequence_measure, sms_2, sms_3):
    with pytest.warns(UserWarning, match="No measures are calculated as there are not any times "
                                         "that match between the two state sequences."):
        assert state_sequence_measure(sms_2, sms_3) == []


def test_state_sequence_measure_bad_time(state_sequence_measure, sms_2, sms_3):
    with pytest.raises(IndexError):
        state_sequence_measure(sms_2, sms_3, [datetime.datetime(2023, 10, 1, 0, 0, 20)])


def test_recent_state_sequence_measure_1(sms_1, sms_3):
    measure = non_state_measures.RecentStateSequenceMeasure(state_measure=measures.Euclidean(),
                                                            n_states_to_compare=2)

    assert measure(sms_1, sms_3) == [7, 8]


def test_recent_state_sequence_measure_2(sms_1, sms_3):
    measure = non_state_measures.RecentStateSequenceMeasure(state_measure=measures.Euclidean(),
                                                            n_states_to_compare=5)

    assert measure(sms_1, sms_3) == [5, 6, 7, 8]


class DummyMultiMeasure(non_state_measures.MultipleMeasure):

    def __call__(self, list_1: list, list_2: list) -> list:
        # Combine the two lists and return them
        combined_list = [*list_1, *list_2]
        return combined_list


@pytest.mark.parametrize("mean_measure_input, expected_mean_measure_output",
                         [
                             ([17, 19, -2, 15, 16], 13),
                             ([], None),
                             ([19, -42, 1.1, 3.14, 9, 7.5, -294.68, 11, 171, 6.3948], -10.85452),
                             ([2.41], 2.41)
                         ])
def test_mean_measure(mean_measure_input, expected_mean_measure_output):
    multi_measure = DummyMultiMeasure()
    mean_measure = non_state_measures.MeanMeasure(measure=multi_measure)

    observed_output = mean_measure(mean_measure_input, [])

    assert observed_output == pytest.approx(expected_mean_measure_output)


@pytest.mark.parametrize("set_comparison_input_1, set_comparison_input_2, "
                         "expected_set_comparison_output",
                         [({1, 2, 3}, {3, 2, 1}, 1.0),  # Check Matching
                          ([1, 2, 3, 2], [1, 2, 3], 1.0),  # Check works with a list
                          ({1, 2, 3}, {4, 5, 6}, 0.0),  # Check no matches
                          ({*"act"}, {*"pact"}, 0.75),  # Check letters
                          ({6, 4, 7, 3, 12}, {14, 11, 2, 12, 4}, 0.25),
                          ])
def test_set_comparison_measure(set_comparison_input_1, set_comparison_input_2,
                                expected_set_comparison_output):
    measure = non_state_measures.SetComparisonMeasure()

    observed_output = measure(set_comparison_input_1, set_comparison_input_2)

    assert observed_output == pytest.approx(expected_set_comparison_output)
