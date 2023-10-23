from typing import Tuple, Collection

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..base import Property
from ..dataassociator.base import Associator
from ..measures import BaseMeasure
from ..types.association import Association, AssociationSet


class OneToOneAssociator(Associator):
    """
    This a general one to one associator. It can be used to associate objects/values that have a
    :class:`~.BaseMeasure` to compare them.
    Uses :func:`~scipy.optimize.linear_sum_assignment` to find the minimum (or maximum) measure by
    combination objects from two sources.

    Notes
    -----
    As default the association threshold is set to +- a large number (1e10 was chosen arbitrarily).
    Infinity can't be used, as it breaks the association algorithm.
    """

    measure: BaseMeasure = Property(
        doc="This will compare two objects that could be associated together and will provide an "
            "indication of the separation between the objects.")
    association_threshold: float = Property(
        default=None,
        doc="The maximum (minimum if :attr:`~.maximise_measure` is true) value from the "
            ":attr:`~.measure` needed to associate two objects. If the default value of `None` is "
            "used then the association threshold is set to plus/minus an arbitrarily large number "
            "that shouldn't limit associations.")

    maximise_measure: bool = Property(
        default=False, doc="Should the association algorithm attempt to maximise or minimise the "
                           "output of the measure.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.association_threshold is None:
            if self.maximise_measure:
                self.association_threshold = -1e10
            else:
                self.association_threshold = 1e10

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
        # to assign tracks to nearest detection
        # Maximise flag = true for probability instance
        # (converts minimisation problem to maximisation problem)
        row_ind, col_ind = linear_sum_assignment(
            distance_matrix, self.maximise_measure)

        # Create dictionary for associations
        associations = AssociationSet()

        # Generate dict of key/value pairs
        for i, j in zip(row_ind, col_ind):
            object_a = list_of_as[i]
            object_b = list_of_bs[j]

            value = distance_matrix[i, j]

            # Check association meets threshold
            if self.maximise_measure:
                if value > self.association_threshold:
                    # Meets threshold
                    associations.associations.add(Association({object_a, object_b}))
            else:  # Minimise measure
                if value < self.association_threshold:
                    # Meets threshold
                    associations.associations.add(Association({object_a, object_b}))

        associated_objects = {obj
                              for assoc in associations.associations
                              for obj in assoc.objects}

        unassociated_a = set(objects_a) - associated_objects
        unassociated_b = set(objects_b) - associated_objects

        return associations, unassociated_a, unassociated_b

    @property
    def fail_value(self):
        """
        For an association to be valid is must be over (or under if maximise_measure is True)
        (non-inclusive). Therefore setting the value to the association threshold will result in
        the association not taking place.
        """
        return self.association_threshold

    def individual_weighting(self, a, b):
        """ This wrapper around the measure function allows for filtering/error checking of the
        measure function. It can give an easy access point for subclasses that want to apply
        additional filtering or gating."""
        measure_output = self.measure(a, b)
        if measure_output is None:
            return self.fail_value
        else:
            if self.maximise_measure:
                return max(measure_output, self.fail_value)
            else:
                return min(measure_output, self.fail_value)

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
