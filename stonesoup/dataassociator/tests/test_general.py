import pytest

from ...measures.base import BaseMeasure
from ..general import OneToOneAssociator


class ProductMeasure(BaseMeasure):
    sign = 1

    def __call__(self, item_a, item_b):
        return self.sign * item_a[1] * item_b[1]


class NegativeProductMeasure(ProductMeasure):
    sign = -1


@pytest.mark.parametrize(
    ("measure", "maximise_measure", "association_threshold"),
    [
        (ProductMeasure(), True, 5),
        (NegativeProductMeasure(), False, -5),
    ],
)
def test_threshold_allows_optimal_partial_assignment(
        measure, maximise_measure, association_threshold):
    objects_a = (("a", 1), ("a", 2), ("a", 3), ("a", 5))
    objects_b = (("b", 1), ("b", 2), ("b", 4), ("b", 7))

    associator = OneToOneAssociator(
        measure=measure,
        maximise_measure=maximise_measure,
        association_threshold=association_threshold,
    )

    associations, unassociated_a, unassociated_b = associator.associate(objects_a, objects_b)

    actual_pairs = {frozenset(association.objects)
                    for association in associations.associations}
    expected_pairs = {
        frozenset((objects_a[1], objects_b[2])),
        frozenset((objects_a[2], objects_b[1])),
        frozenset((objects_a[3], objects_b[3])),
    }

    assert actual_pairs == expected_pairs
    assert unassociated_a == {objects_a[0]}
    assert unassociated_b == {objects_b[0]}
