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
        default=float('nan'),
        doc="For an association to be valid is must be over (or under if maximise_measure is True)"
            " (non-inclusive). This is the value given to associations beyond the threshold."
            "Infinite values are assigned this value")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_non_association_cost()
        if self.non_association_cost is None:
            warnings.warn("Setting `non_association_cost` to None means that no preprocessing "
                          "takes place on the distance matrix. This can lead to unexpected "
                          "behaviour and is not recommended.")

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

        original_distance_matrix = np.empty((len(objects_a), len(objects_b)))

        list_of_as = list(objects_a)
        list_of_bs = list(objects_b)

        # Calculate the measure for each combination of objects
        for i, a in enumerate(list_of_as):
            for j, b in enumerate(list_of_bs):
                original_distance_matrix[i, j] = self.measure(a, b)

        distance_matrix = self.scale_distance_matrix(original_distance_matrix)

        print(distance_matrix)

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

            value = original_distance_matrix[i, j]

            # Check association meets threshold
            if self.is_measure_valid(value):
                associations.associations.add(Association(OrderedSet([object_a, object_b])))
                unassociated_a.discard(object_a)
                unassociated_b.discard(object_b)

        return associations, unassociated_a, unassociated_b

    def scale_distance_matrix(self, input_matrix: np.ndarray) -> np.ndarray:


        max_value = self.get_maximum_value(input_matrix)

        x = input_matrix.copy()

        # Scale infinite values to the maximum value
        x[np.isinf(x)] = max_value * np.sign(x[np.isinf(x)])

        if self.non_association_cost is not None:
            # outside_threshold_mask = np.array([
            #     not self.is_measure_within_association_threshold(value)
            #     for value in input_matrix
            # ])
            # x[outside_threshold_mask] = self.non_association_cost
            inside_threshold_mask = np.vectorize(self.is_measure_within_association_threshold)(input_matrix)
            x[~inside_threshold_mask] = self.non_association_cost

        if self.maximise_measure:
            x[np.isnan(x) | np.isnan(input_matrix)] = -max_value
        else:
            x[np.isnan(x) | np.isnan(input_matrix)] = max_value

        return x

    @staticmethod
    def get_maximum_value(a: np.ndarray) -> float:
        "This value should be larger than any output from the measure function. It should also"
        " be small enough that ``min_measure + measure_fail_value > measure_fail_value``. For "
        "example `` 1e12 + 1e-12 == 1e12``. The default value (1e10) is suitable for measure "
        "outputs between 1e-6 and 1e10."

        finite_a = a[np.isfinite(a) & (a != 0)]
        if len(finite_a) == 0:
            max_value = 1e10
            return max_value

        max_a = np.max(np.abs(finite_a))
        min_a = np.min(np.abs(finite_a))

        for multi_factor in [1e6, 1e5, 1e4, 1e3, 1e2]:
            max_value = multi_factor * max_a

            if max_value + min_a > max_value:
                break

        if max_value + min_a == max_value:
            warnings.warn("Maximum value is too high. Precision will lost in addition. "
                          f"max_value({max_value}) + min_value({min_a}) == max_value")

        return max_value


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
