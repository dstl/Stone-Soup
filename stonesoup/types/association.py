import datetime
from typing import Set, Union
from itertools import combinations

from ..base import Property
from .base import Type
from .time import TimeRange, CompoundTimeRange


class Association(Type):
    """Association type

    An association between objects
    """

    # TODO: Should probably add a link to the associator that produced it
    objects: Set = Property(doc="Set of objects being associated")


class AssociationPair(Association):
    """AssociationPair type

    An :class:`~.Association` representing the association of two objects
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self.objects) != 2:
            raise ValueError("Only two objects can be associated in one "
                             "AssociationPair object")


class SingleTimeAssociation(Association):
    """SingleTimeAssociation type

    An :class:`~.Association` representing the linking of objects at a single
    time
    """

    timestamp: datetime.datetime = Property(
        default=None,
        doc="Timestamp of the association. Default is None.")


class TimeRangeAssociation(Association):
    """TimeRangeAssociation type

    An :class:`~.AssociationPair` representing the linking of objects over a
    range of times
    """

    time_range: Union[CompoundTimeRange, TimeRange] = Property(
        default=None, doc="Range of times that association exists over. Default is None")


class AssociationSet(Type):
    """AssociationSet type

    A set of :class:`~.Association` type objects representing multiple
    independent associations. Contains functions for indexing into the
    associations
    """

    associations: Set[Association] = Property(default=None, doc="Set of independent associations")

    def __init__(self, associations=None, *args, **kwargs):
        super().__init__(associations, *args, **kwargs)
        if self.associations is None:
            self.associations = set()
        if not all(isinstance(member, Association) for member in self.associations):
            raise TypeError("Association set must contain only Association instances")
        self._simplify()

    def __eq__(self, other):
        return self.associations == other.associations

    def add(self, association):
        if association is None:
            return
        elif isinstance(association, Association):
            self.associations.add(association)
        elif isinstance(association, AssociationSet):
            for component in association:
                self.add(component)
        else:
            raise TypeError("Supplied parameter must be an Association or AssociationSet")
        self._simplify()

    def _simplify(self):
        """Where multiple associations describe the same pair of objects, combine them into one.
        Note this is only implemented for pairs with a time_range attribute - others will be skipped
        """
        to_remove = set()
        for (assoc1, assoc2) in combinations(self.associations, 2):
            if not (len(assoc1.objects) == 2 and len(assoc2.objects) == 2) or \
                    not(hasattr(assoc1, 'time_range') and hasattr(assoc2, 'time_range')):
                continue
            if assoc1.objects == assoc2.objects:
                if isinstance(assoc1.time_range, CompoundTimeRange):
                    assoc1.time_range.add(assoc2.time_range)
                    to_remove.add(assoc2)
                elif isinstance(assoc2.time_range, CompoundTimeRange):
                    assoc2.time_range.add(assoc1.time_range)
                    to_remove.add(assoc1)
                else:
                    assoc1.time_range = CompoundTimeRange([assoc1.time_range, assoc2.time_range])
                    to_remove.add(assoc2)
        for assoc in to_remove:
            self.remove(assoc)

    def remove(self, association):
        if association is None:
            return
        elif isinstance(association, Association):
            if association not in self.associations:
                raise ValueError("Supplied parameter must be contained by this instance")
            self.associations.remove(association)
        elif isinstance(association, AssociationSet):
            for component in association:
                self.remove(component)
        else:
            raise TypeError("Supplied parameter must be an Association or AssociationSet")

    @property
    def key_times(self):
        """Returns all timestamps at which a component starts or ends, or where there is a
        :class:`.~SingleTimeAssociation`."""
        key_times = list(self.overall_time_range.key_times)
        for association in self.associations:
            if isinstance(association, SingleTimeAssociation):
                key_times.append(association.timestamp)
        return sorted(key_times)

    @property
    def overall_time_range(self):
        """Returns a :class:`~.CompoundTimeRange` covering all times at which at least
        one association is active.

        Note: :class:`~.SingleTimeAssociation` are not counted
        """
        overall_range = CompoundTimeRange()
        for association in self.associations:
            if hasattr(association, 'time_range'):
                overall_range.add(association.time_range)
        return overall_range

    @property
    def object_set(self):
        """Returns a set of all objects contained by this instance.
        """
        object_set = set()
        for assoc in self.associations:
            for obj in assoc.objects:
                object_set.add(obj)
        return object_set

    def associations_at_timestamp(self, timestamp):
        """Return the associations that exist at a given timestamp

        Method will return a set of all the  :class:`~.Association` type
        objects which occur at the specified time stamp.

        Parameters
        ----------
        timestamp: datetime.datetime
            Timestamp at which associations should be identified

        Returns
        -------
        : :class:`~.AssociationSet`
            Associations which occur at specified timestamp
        """
        if not isinstance(timestamp, datetime.datetime):
            raise TypeError("Supplied parameter must be a datetime.datetime object")
        ret_associations = set()
        for association in self.associations:
            # If the association is at a single time
            if hasattr(association, "timestamp"):
                if association.timestamp == timestamp:
                    ret_associations.add(association)
            else:
                if timestamp in association.time_range:
                    ret_associations.add(association)
        return AssociationSet(ret_associations)

    def associations_including_objects(self, objects):
        """Return associations that include all the given objects

        Method will return the set of all the :class:`~.Association` type
        objects which contain an association with the provided object

        Parameters
        ----------
        objects: set of objects
            Set of objects to look for in associations
        Returns
        -------
        : class:`~.AssociationSet`
            A set of associations containing every member of objects
        """
        # Ensure objects is iterable
        if not isinstance(objects, list) and not isinstance(objects, set):
            objects = {objects}

        return AssociationSet({association
                              for association in self.associations
                              if all(object_ in association.objects
                                     for object_ in objects)})

    def __contains__(self, item):
        return item in self.associations

    def __iter__(self):
        return iter(self.associations)

    def __len__(self):
        return len(self.associations)
