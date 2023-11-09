import pytest

from stonesoup.measures import base as base_measures


def test_track_measure_raise_error():
    with pytest.raises(TypeError):
        base_measures.TrackMeasure()


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
    measure = base_measures.SetComparisonMeasure()

    observed_output = measure(set_comparison_input_1, set_comparison_input_2)

    assert observed_output == pytest.approx(expected_set_comparison_output)
