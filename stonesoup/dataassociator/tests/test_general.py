import math
import warnings
from typing import Collection, Optional

import numpy as np
import pytest

from stonesoup.dataassociator.general import GreedyAssociator, OneToOneAssociator, \
    GeneralAssociator
from stonesoup.measures.base import BaseMeasure


def check_associator(associator: GeneralAssociator,
                     objs_a: Collection, objs_b: Collection,
                     expected_associations: dict,
                     expected_unassociated_a: set, expected_unassociated_b: set):

    associations, unassociated_a, unassociated_b = associator.associate(objs_a, objs_b)

    assoc_dict = {association.objects[0]: association.objects[1]
                  for association in associations}

    assert assoc_dict == expected_associations, f"\n" \
                                                f"\tActual dict: {assoc_dict}\n" \
                                                f"\tExpected dict{expected_associations}"
    assert unassociated_a == expected_unassociated_a
    assert unassociated_b == expected_unassociated_b


class NumericDifference(BaseMeasure):

    def __call__(self, item1: float, item2: float) -> float:
        """ Returns absolute difference between two floats"""
        return abs(item1 - item2)


@pytest.mark.parametrize(
    "association_threshold, maximise_measure, objs_a, objs_b, expected_associations, "
    "expected_unassociated_a, expected_unassociated_b",
    [
        (None, True, (1, 2, 3, 4), (8, 9, 10), {1: 10, 2: 10, 3: 10, 4: 10},
         set(), {8, 9}),
        (None, False, (3, 11, 12, 1000), (5, 7, 41, 42), {3: 5, 11: 7, 12: 7, 1000: 42},
         set(), {41}),
        (10, False, (3, 11, 12, 30, 52, 1000), (5, 7, 41, 42), {3: 5, 11: 7, 12: 7, 52: 42},
         {30, 1000}, {41}),
        (7, True, (1, 2, 3, 4), (8, 9, 10), {1: 10, 2: 10, 3: 10},
         {4}, {8, 9})
    ]
)
def test_greedy_associator(association_threshold: float, maximise_measure: bool,
                           objs_a: Collection, objs_b: Collection, expected_associations: dict,
                           expected_unassociated_a: set, expected_unassociated_b: set):

    associator = GreedyAssociator(measure=NumericDifference(),
                                  association_threshold=association_threshold,
                                  maximise_measure=maximise_measure)

    return check_associator(associator, objs_a, objs_b, expected_associations,
                            expected_unassociated_a, expected_unassociated_b)


class SquareNumericDifference(BaseMeasure):

    def __call__(self, item1: float, item2: float) -> float:
        """ Returns absolute difference between two floats"""
        return (item1-item2)**2


@pytest.mark.parametrize(
    "association_threshold, non_association_cost, expected_associations, "
    "expected_unassociated_a, expected_unassociated_b",
    [(None, None, {0: 8, 2: 7, 5: 6}, {7}, set()),  # Total measure: 90
     (5, None, {0: 8, 2: 7}, {5, 7}, {6}),  # Total measure: 89
     (5, float('nan'), {0: 7, 2: 6, 5: 8}, {7}, set()),  # Total measure: 74
     (5, None, {0: 8, 2: 7}, {5, 7}, {6}),  # Total measure: 89
     ]
)
def test_one_to_one_associator_max(association_threshold: Optional[float],
                                   non_association_cost: Optional[float],
                                   expected_associations: dict,
                                   expected_unassociated_a: set, expected_unassociated_b: set):

    measure = SquareNumericDifference()
    maximise_measure = True

    objs_a = (0, 2, 5, 7)
    objs_b = (6, 7, 8)

    associator = OneToOneAssociator(measure=measure,
                                    association_threshold=association_threshold,
                                    maximise_measure=maximise_measure,
                                    non_association_cost=non_association_cost)

    return check_associator(associator, objs_a, objs_b, expected_associations,
                            expected_unassociated_a, expected_unassociated_b)


class MultiplyMeasure(BaseMeasure):
    def __call__(self, item1: float, item2: float) -> float:
        return item1*item2


# test_threshold_in_one_to_one_associator and test_one_to_one_associator_min use
# ``objs_a = (2, 3, 5, 8)`` and ``objs_b = (1, 4, 7, 10)``. Below is multiplication table for the
# numbers to aid understanding
#
#    |   2   3   5   8
# ---+----------------
#  1 |   2   3   5   8
#  2 |   8  12  20  32
#  7 |  14  21  35  56
# 10 |  20  30  50  80
@pytest.mark.parametrize(
    "association_threshold, expected_associations, "
    "expected_unassociated_a, expected_unassociated_b",
    [(None, {2: 10, 3: 7, 5: 4, 8: 1}, set(), set()),  # Everything is within threshold
     (21, {2: 10, 3: 7, 5: 4, 8: 1}, set(), set()),  # Associations are within threshold
     (20, {2: 10, 5: 4, 8: 1}, {3}, {7}),  # 3: 7 is no longer within threshold
     (19, {8: 1}, {3, 2, 5}, {7, 10, 4}),  # 2: 10 and 5: 4 are no longer within threshold
     (8, {8: 1}, {3, 2, 5}, {7, 10, 4}),
     (7, dict(), {3, 2, 5, 8}, {7, 10, 4, 1}),  # 8: 1 is no longer within threshold
     ]
)
def test_threshold_in_one_to_one_associator(association_threshold: Optional[float],
                                            expected_associations: dict,
                                            expected_unassociated_a: set,
                                            expected_unassociated_b: set):

    measure = MultiplyMeasure()
    maximise_measure = False
    non_association_cost = None

    objs_a = (2, 3, 5, 8)
    objs_b = (1, 4, 7, 10)

    associator = OneToOneAssociator(measure=measure,
                                    association_threshold=association_threshold,
                                    maximise_measure=maximise_measure,
                                    non_association_cost=non_association_cost)

    return check_associator(associator, objs_a, objs_b, expected_associations,
                            expected_unassociated_a, expected_unassociated_b)


@pytest.mark.parametrize(
    "non_association_cost, expected_associations, "
    "expected_unassociated_a, expected_unassociated_b",
    [(None, {2: 10, 3: 7, 5: 4, 8: 1}, set(), set()),  # Total measure: 69
     (22, {2: 7, 3: 4, 5: 1}, {8}, {10}),  # Total measure: 53 = 31 + 22 (non_association_cost)
     (25, {2: 7, 3: 4, 5: 1}, {8}, {10}),  # Total measure: 56 = 31 + 25 (non_association_cost)
     (30, {2: 7, 3: 4, 5: 1}, {8}, {10}),  # Total measure: 61 = 31 + 30 (non_association_cost)
     (50, {2: 10, 3: 7, 5: 4, 8: 1}, set(), set()),  # Total measure: 69
     (float('nan'), {2: 10, 3: 7, 5: 4, 8: 1}, set(), set()),  # Total measure: 69
     ]
)
def test_one_to_one_associator_min(non_association_cost: Optional[float],
                                   expected_associations: dict,
                                   expected_unassociated_a: set, expected_unassociated_b: set):

    measure = MultiplyMeasure()
    association_threshold = 21
    maximise_measure = False

    objs_a = (2, 3, 5, 8)
    objs_b = (1, 4, 7, 10)

    associator = OneToOneAssociator(measure=measure,
                                    association_threshold=association_threshold,
                                    maximise_measure=maximise_measure,
                                    non_association_cost=non_association_cost)

    return check_associator(associator, objs_a, objs_b, expected_associations,
                            expected_unassociated_a, expected_unassociated_b)


@pytest.mark.parametrize(
    "maximise_measure, association_threshold, non_association_cost",
    [(True, 10, 11),
     (False, 10, 9)]
)
def test_non_association_cost_warning(maximise_measure: bool,
                                      association_threshold: Optional[float],
                                      non_association_cost: Optional[float]):

    with pytest.warns(UserWarning):
        OneToOneAssociator(measure=None,
                           association_threshold=association_threshold,
                           maximise_measure=maximise_measure,
                           non_association_cost=non_association_cost)


@pytest.mark.parametrize('associator_class', [OneToOneAssociator, GreedyAssociator])
@pytest.mark.parametrize("association_threshold", [None, 10])
def test_accept_special_values(associator_class: type, association_threshold: Optional[float]):
    def add_measure(a, b):
        return a+b

    associator = associator_class(measure=add_measure,
                                  maximise_measure=True,
                                  association_threshold=association_threshold)

    objs_a = (float('nan'), float("inf"), float("-inf"))
    objs_b = [True for _ in objs_a]

    # check_associator doesn't work with nan values
    associations, unassociated_a, unassociated_b = associator.associate(objs_a, objs_b)

    assoc_dict = {association.objects[0]: True
                  for association in associations}

    for special_value in unassociated_a:
        assoc_dict[special_value] = False

    for special_value, is_associated in assoc_dict.items():
        if math.isnan(special_value):
            assert is_associated is False

        elif special_value == float('inf'):
            assert is_associated is True

        elif special_value == float('-inf'):
            if association_threshold is None:
                assert is_associated is True
            else:
                assert is_associated is False
        else:
            raise ValueError(f"No logic available for '{special_value}'.")


@pytest.mark.parametrize(
    "input_matrix",
    [
        np.array([1, 2, 3]),  # Test ints
        np.array([1e8, 2e9, 3e10]),  # Test large numbers
        np.array([1e18, 2e19, 3e20]),  # Test very large numbers
        np.array([1e-8, 2e-9, 3e-10]),  # Test small numbers
        np.array([1e-18, 2e-19, 3e-20]),  # Test very small numbers
        np.array([1e-3, 2e0, 3e3]),  # Test small range of numbers
        np.array([1e-6, 2e-2, 3e4]),  # Test medium range of numbers
        np.array([1e-10, 2e2, 3e10]),  # Test large range of numbers
        np.array([1e-20, 2e2, 3e20]),  # Test very large range of numbers (likely to be a warning)
    ]
)
def test_max_allowed_value(input_matrix: np.ndarray):
    max_input = max(input_matrix)
    min_input = min(input_matrix)

    with pytest.warns() as warning_record:
        warnings.warn("Placeholder")
        max_value = OneToOneAssociator.get_maximum_value(input_matrix)

    # `max_value` must be significantly higher than the maximum value
    assert max_value >= max_input * 100

    # `max_value` should small enough that max_value + min(input_matrix) > max_value.
    if max_value + min_input > max_value:
        pass
    else:
        assert len(warning_record) == 2


# Variables for `test_matrix_adjust` (tma) test function
tma_input_matrix = np.array([0, 10, 1e6, -1e6, float('inf'), - float('inf'), float('nan')])
tma_max_value = OneToOneAssociator.get_maximum_value(tma_input_matrix)

@pytest.mark.parametrize(
    "associator, expected_output",
    [
        (  # One
                OneToOneAssociator(measure=None, maximise_measure=False,
                                   association_threshold=None, non_association_cost=None),
                [0, 10, 1e6, -1e6, tma_max_value, - tma_max_value, tma_max_value]),
        (  # Two
                OneToOneAssociator(measure=None, maximise_measure=False,
                                   association_threshold=None, non_association_cost=np.nan),
                [0, 10, 1e6, -1e6, tma_max_value, - tma_max_value, tma_max_value]),
        (  # Three
                OneToOneAssociator(measure=None, maximise_measure=False,
                                   association_threshold=5, non_association_cost=np.nan),
                [0, tma_max_value, tma_max_value, -1e6, tma_max_value, - tma_max_value,
                 tma_max_value]),
        (  # Four
                OneToOneAssociator(measure=None, maximise_measure=True,
                                   association_threshold=5, non_association_cost=-1),
                [-1, 10, 1e6, -1, tma_max_value, -1, - tma_max_value]),
        (  # Five
                OneToOneAssociator(measure=None, maximise_measure=False,
                                   association_threshold=10, non_association_cost=20),
                [0, 10, 20, -1e6, 20, -tma_max_value, tma_max_value]),
        (  # Six
                OneToOneAssociator(measure=None, maximise_measure=False,
                                   association_threshold=None, non_association_cost=20),
                [0, 10, 1e6, -1e6, tma_max_value, -tma_max_value, tma_max_value]),

    ]
)
def test_matrix_adjust(associator, expected_output):
    adjusted_matrix = associator.scale_distance_matrix(tma_input_matrix)
    np.testing.assert_equal(adjusted_matrix, expected_output)
