# -*- coding: utf-8 -*-
import datetime

from ..base import Property
from .base import Type


class Association(Type):
    """Association type

    An association between objects
    """

    objects = Property(set,
                       doc="Set of objects being associated")
    # Should probably add a link to the associator that produced it

class AssociationPair(Association):
    """AssociationPair type

    An :class:`~.Association` representing the association of two objects"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self.objects) != 2:
            raise ValueError("Only two objects can be associated in one "
                             "AssociationPair object")

class SingleTimeAssociation(AssociationPair):
    """SingleTimeAssociation type

     An :class:`~.AssociationPair` representing the linking of two objects
     at a single time"""

    timestamp = Property(datetime.datetime, default=None,
                         doc="Timestamp of the association. Default None.")

class TimePeriodAssociation(AssociationPair):
    """TimePeriodAssociation type

     An :class:`~.AssociationPair` representing the linking of two objects
     over a range of times"""

    start_timestamp = Property(datetime.datetime, default=None,
                               doc = "Time that the association begins at")
    end_timestamp = Property(datetime.datetime, default = None,
                             doc="Time that the association ends at")

class AssociationSet(Type):
    """AssociationSet type

    A set of :class:`~.Association` type objects representing multiple
    independent associations. Contains functions for indexing into the
    associations"""

    associations = Property(set, default = set(),
                            doc="Set of independant associations")

    def associations_at_timestamp(self,timestamp):
        "Return the assocations that exist at a given timestamp"
        ret_associations = set()
        for association in self.associations:
            "If the association is at a single time"
            if hasattr(association,"timestamp"):
                if association.timestamp == timestamp:
                    ret_associations.add(association)
            else:
                if (timestamp >= association.start_timestamp and
                            timestamp <= association.end_timestamp):

                    ret_associations.add(association)
        return ret_associations

    def associations_including_objects(self,objects):
        "Return associations that include all the given objects"
        ret_associations = set()
        for association in self.associations:

            for obj in objects:
                if obj not in association.objects:
                    continue
            ret_associations.add(association)
        return ret_associations

