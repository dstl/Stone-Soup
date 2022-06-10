import datetime
from typing import Set

from ..base import Property
from .base import Type
from .time import CompoundTimeRange


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

    time_range: CompoundTimeRange = Property(
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
        : set of :class:`~.Association`
            Associations which occur at specified timestamp
        """
        ret_associations = set()
        for association in self.associations:
            # If the association is at a single time
            if hasattr(association, "timestamp"):
                if association.timestamp == timestamp:
                    ret_associations.add(association)
            else:
                if timestamp in association.time_range:
                    ret_associations.add(association)
        return ret_associations

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
        : set of :class:`~.Association`
            A set of objects which have been associated
        """

        # Ensure objects is iterable
        if not isinstance(objects, list) and not isinstance(objects, set):
            objects = {objects}

        return {association
                for association in self.associations
                for object_ in objects
                if object_ in association.objects}

    def get_key_times(self):
        """Return all times at which an association from the set begins or ends

        This method will return an ordered list of the times at which an association in the set
        begins or ends.  Note that in the case of a :class:`~.CompoundTimeRange`,
        there are potentially multiple start and end times,
        and for :class:`~.SingleTimeAssociation` associations, the start and end time are the same

        Returns
        -------
        : list of :class:`datetime.datetime`
            A list of times at which an association starts or ends
        """
        key_times = []
        for association in self.associations:
            if isinstance(association, SingleTimeAssociation):
                key_times.append(association.timestamp)
            else:
                key_times.extend(association.time_range.get_key_times())

        return key_times


    def __contains__(self, item):
        return item in self.associations

    def __iter__(self):
        return iter(self.associations)

    def __len__(self):
        return len(self.associations)
