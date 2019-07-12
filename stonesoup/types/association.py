# -*- coding: utf-8 -*-
import datetime

from ..base import Property
from .base import Type
from .time import TimeRange


class Association(Type):
    """Association type

    An association between objects
    """

    # TODO: Should probably add a link to the associator that produced it
    objects = Property(set, doc="Set of objects being associated")


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

    timestamp = Property(datetime.datetime, default=None,
                         doc="Timestamp of the association. Default is None.")


class TimeRangeAssociation(Association):
    """TimeRangeAssociation type

     An :class:`~.AssociationPair` representing the linking of objects over a
    range of times
    """

    time_range = Property(TimeRange, default=None,
                          doc="Range of times that association exists over. "
                              "Default is None")


class AssociationSet(Type):
    """AssociationSet type

    A set of :class:`~.Association` type objects representing multiple
    independent associations. Contains functions for indexing into the
    associations
    """

    associations = Property(set, default=None,
                            doc="Set of independant associations")

    def __init__(self, associations=None, *args, **kwargs):
        if associations is None:
            associations = set()
        super().__init__(associations, *args, **kwargs)

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

        Paramters
        ---------
        : objects:
            Set of objects
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
