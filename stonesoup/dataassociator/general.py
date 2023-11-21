import warnings
from abc import abstractmethod
from typing import Tuple, Collection, Callable, Union, Optional

import numpy as np
from ordered_set import OrderedSet
from scipy.optimize import linear_sum_assignment

from ..base import Property
from ..dataassociator.base import Associator
from ..measures.base import BaseMeasure
from ..types.association import Association, AssociationSet


class GeneralAssociator(Associator):
    """ This a general associator base class."""

    measure: Union[BaseMeasure, Callable] = Property(
        doc="This will compare two objects that could be associated together and will provide an "
            "indication of the separation between the objects.")
    association_threshold: Optional[float] = Property(
        default=None,
        doc="The maximum (minimum if :attr:`~.maximise_measure` is true) value from the "
            ":attr:`~.measure` needed to associate two objects. A value of ``None`` means no "
            "threshold is applied. This is the default option")

    maximise_measure: bool = Property(
        default=False, doc="Should the association algorithm attempt to maximise or minimise the "
                           "output of the measure.")

    def is_measure_valid(self, value: float) -> bool:
        """Is the output the `measure` valid."""
        if np.isnan(value):
            return False
        else:
            return self.is_measure_within_association_threshold(value)

    def is_measure_within_association_threshold(self, value: float) -> bool:

        if self.association_threshold is None:
            return True

        if self.maximise_measure:
            return self.association_threshold <= value
        else:  # Minimise measure
            return self.association_threshold >= value

    def association_dict(self, objects_a: Collection, objects_b: Collection) -> dict:
        """
        This is a wrapper function around the :meth:`~.associate` function. The two collections of
        objects are associated to each other. The objects are entered into a dictionary:

        * The dictionary key is an object from either collection.
        * The value is the object it is associated to. If the key object isn't associated to an
          object then the value is `None`.

        As the objects are used as dictionary keys, they must be hashable or a :class:`~.TypeError`
        will be raised.

        Parameters
        ----------
        objects_a : collection of hashable objects to associate to the objects in ``objects_b``
        objects_b : collection of hashable objects to associate to the objects in ``objects_a``

        Returns
        -------
        AssociationSet
            Contains a set of :class:`~.Association` objects

        """
        output_dict = {}
        associations, unassociated_a, unassociated_b = self.associate(objects_a, objects_b)

        for assoc in associations.associations:
            object_1, object_2 = assoc.objects
            output_dict[object_1] = object_2
            output_dict[object_2] = object_1

        for obj in [*unassociated_a, *unassociated_b]:
            output_dict[obj] = None

        return output_dict

    @abstractmethod
    def associate(self, objects_a: Collection, objects_b: Collection) \
            -> Tuple[AssociationSet, Collection, Collection]:
        """Associate two collections of objects together.

        Parameters
        ----------
        objects_a : collection of objects to associate to the objects in `objects_b`
        objects_b : collection of objects to associate to the objects in `objects_a`

        Returns
        -------
        AssociationSet
            Contains a set of :class:`~.Association` objects

        """


class OneToOneAssociator(GeneralAssociator):
    """
    This a general one to one associator. It can be used to associate objects/values that have a
    :class:`~.BaseMeasure` to compare them.
    Uses :func:`~scipy.optimize.linear_sum_assignment` to find the minimum (or maximum) measure by
    combination objects from two sources.

    Notes
    -----
    """

    non_association_cost: Optional[float] = Property(
        default=None,
        doc="For an association to be valid is must be over (or under if maximise_measure is True)"
            " (non-inclusive). This is the value given to associations above the threshold.")

    measure_fail_magnitude: float = Property(
        default=1e10,
        doc="This value should be larger than any output from the measure function. It should also"
            " be small enough that ``min_measure + measure_fail_value > measure_fail_value``. For "
            "example `` 1e12 + 1e-12 == 1e12``. The default value (1e10) is suitable for measure "
            "outputs between 1e-6 and 1e10."
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_non_association_cost()

    @property
    def measure_fail_value(self) -> float:
        """:func:`~scipy.optimize.linear_sum_assignment` cannot function with `nan` values. This
        is the value to replace `nan` for the input into *linear_sum_assignment*."""
        if self.maximise_measure:
            return -self.measure_fail_magnitude
        else:
            return self.measure_fail_magnitude

    def check_non_association_cost(self):
        """
        Check is the `non_association_cost` is valid and above/below the `association_threshold`.
        """

        if self.association_threshold is not None and \
                self.non_association_cost is not None:

            if self.maximise_measure:
                if self.association_threshold <= self.non_association_cost:
                    warnings.warn(
                        f"When trying to minimise the measure the non_association_cost "
                        f"({self.non_association_cost}) should be larger than the "
                        f"association_threshold ({self.association_threshold}).")
            else:
                if self.association_threshold >= self.non_association_cost:
                    warnings.warn(
                        f"When trying to maximise the measure the non_association_cost "
                        f"({self.non_association_cost}) should be smaller than the "
                        f"association_threshold ({self.association_threshold}).")

    def apply_non_association_cost(self, value: float) -> float:
        """Replace the `value` with the `non_association_cost` if necessary."""

        # Can't apply non-association cost to a non-match
        if np.isnan(value):
            return value

        # There isn't a non association cost
        if self.non_association_cost is None:
            return value

        # non association cost doesn't need to be applied
        if self.is_measure_within_association_threshold(value):
            return value
        else:
            return self.non_association_cost

    def individual_weighting(self, a, b) -> float:
        """ This wrapper around the measure function allows for filtering/error checking of the
        measure function. It can give an easy access point for subclasses that want to apply
        additional filtering or gating."""

        measure_output = self.measure(a, b)

        if abs(measure_output) > self.measure_fail_magnitude:
            warnings.warn(f"measure_output ({measure_output}) is larger than the "
                          f"measure_fail_magnitude {self.measure_fail_magnitude}. Increase "
                          f"measure_fail_magnitude to avoid unexpected behaviour.")

        # Check if non_association_cost should be applied
        measure_output = self.apply_non_association_cost(measure_output)

        # Check for non-valid inputs into `linear_sum_assignment` function
        if np.isnan(measure_output):
            filtered_measure_output = self.measure_fail_value
        elif measure_output == float('inf'):
            filtered_measure_output = self.measure_fail_magnitude * 0.999
        elif measure_output == float('-inf'):
            filtered_measure_output = -self.measure_fail_magnitude * 0.999
        else:
            filtered_measure_output = measure_output

        return filtered_measure_output

    def associate(self, objects_a: Collection, objects_b: Collection) \
            -> Tuple[AssociationSet, Collection, Collection]:
        """Associate two collections of objects together. Calculate the measure between each
        object. :func:`~scipy.optimize.linear_sum_assignment` is used to find
        the minimum (or maximum) measure by combination objects from two sources.

        Parameters
        ----------
        objects_a : collection of objects to associate to the objects in `objects_b`
        objects_b : collection of objects to associate to the objects in `objects_a`

        Returns
        -------
        AssociationSet
            Contains a set of :class:`~.Association` objects

        """

        if len(objects_a) == 0 or len(objects_b) == 0:
            return AssociationSet(), objects_a, objects_b

        distance_matrix = np.empty((len(objects_a), len(objects_b)))

        list_of_as = list(objects_a)
        list_of_bs = list(objects_b)

        # Calculate the measure for each combination of objects
        for i, a in enumerate(list_of_as):
            for j, b in enumerate(list_of_bs):
                distance_matrix[i, j] = self.individual_weighting(a, b)

        # Use "shortest path" assignment algorithm on distance matrix
        # to assign object_a to the nearest object_b
        # Maximise flag = true for probability instance
        # (converts minimisation problem to maximisation problem)
        row_ind, col_ind = linear_sum_assignment(
            distance_matrix, self.maximise_measure)

        # Create dictionary for associations
        associations = AssociationSet()

        unassociated_a = set(objects_a)
        unassociated_b = set(objects_b)

        # Generate dict of key/value pairs
        for i, j in zip(row_ind, col_ind):
            object_a = list_of_as[i]
            object_b = list_of_bs[j]

            value = distance_matrix[i, j]

            # Check association meets threshold
            if self.is_measure_valid(value) and value != self.measure_fail_value:
                associations.associations.add(Association(OrderedSet([object_a, object_b])))
                unassociated_a.discard(object_a)
                unassociated_b.discard(object_b)

        return associations, unassociated_a, unassociated_b


class GreedyAssociator(GeneralAssociator):

    def associate(self, objects_a: Collection, objects_b: Collection) \
            -> Tuple[AssociationSet, Collection, Collection]:
        """
        For each object in `objects_a` the locally best object in `objects_b` will be
        associated to it.

        Parameters
        ----------
        objects_a : collection of objects to associate to the objects in `objects_b`
        objects_b : collection of objects to associate to the objects in `objects_a`

        Returns
        -------
        AssociationSet
            Contains a set of :class:`~.Association` objects

        """

        if len(objects_a) == 0 or len(objects_b) == 0:
            return AssociationSet(), objects_a, objects_b

        associations = AssociationSet()

        for obj_a in objects_a:
            association_options = [(self.measure(obj_a, obj_b), obj_b)
                                   for obj_b in objects_b]

            if self.maximise_measure:
                best_association = max(association_options, key=lambda x: x[0])
            else:
                best_association = min(association_options, key=lambda x: x[0])

            best_measure_value, best_obj_b = best_association

            if self.is_measure_valid(best_measure_value):
                associations.associations.add(Association(OrderedSet((obj_a, best_obj_b))))

        associated_objects = {obj
                              for assoc in associations.associations
                              for obj in assoc.objects}

        unassociated_a = set(objects_a) - associated_objects
        unassociated_b = set(objects_b) - associated_objects

        return associations, unassociated_a, unassociated_b
