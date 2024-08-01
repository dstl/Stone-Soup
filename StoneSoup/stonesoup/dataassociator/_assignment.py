import copy
from itertools import islice

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..types.association import AssociationSet


def multidimensional_deconfliction(association_set):
    """Solves the Multidimensional Assignment Problem (MAP)

    The assignment problem becomes more complex when time is added as a dimension.
    This basic solution finds all the conflicts in an association set and then creates a
    matrix of sums of conflicts in seconds, which is then passed to linear_sum_assignment to
    solve as a simple 2D assignment problem.
    Therefore, each object will only ever be assigned to one other
    at any one time.  In the case of an association that only partially overlaps, the time range
    of the "weaker" one (the one eliminated by assign2D) will be trimmed
    until there is no conflict.

    Due to the possibility of more than two conflicting associations at the same time,
    this algorithm is recursive, but it is not expected many (if any) recursions will be required
    for most uses.

    Parameters
    ----------
    association_set: The :class:`AssociationSet` to de-conflict


    Returns
    -------
    : :class:`AssociationSet`
        The association set without contradictory associations
    """
    if check_if_no_conflicts(association_set):
        return copy.copy(association_set)

    objects = list(association_set.object_set)
    length = len(objects)
    totals = np.zeros((length, length))  # Time objects i and j are associated for in total

    for association in association_set.associations:
        if len(association.objects) != 2:
            raise ValueError("Supplied set must only contain pairs of associated objects")
        i, j = (objects.index(object_) for object_ in association.objects)
        totals[i, j] = association.time_range.duration.total_seconds()
    totals = np.maximum(totals, totals.transpose())  # make symmetric

    totals = np.rint(totals).astype(int)
    np.fill_diagonal(totals, 0)  # Don't want to count associations of an object with itself
    solved_2d = linear_sum_assignment(totals, maximize=True)[1]
    cleaned_set = AssociationSet()
    association_set_reduced = copy.copy(association_set)
    for i, j in enumerate(solved_2d):
        if i == j:
            # Can't associate with self
            continue
        try:
            assoc = next(iter(association_set_reduced.associations_including_objects(
                {objects[i], objects[j]})))  # There should only be 1 association in this set
        except StopIteration:
            # We took the association out previously in the loop
            continue
        if all(assoc.duration > clean_assoc.duration or not conflicts(assoc, clean_assoc)
               for clean_assoc in cleaned_set):
            cleaned_set.add(copy.copy(assoc))
            association_set_reduced.remove(assoc)

    if len(cleaned_set) == 0:
        raise ValueError("Problem unsolvable using this method")

    if len(association_set_reduced) == 0:
        if check_if_no_conflicts(cleaned_set):
            raise RuntimeError("Conflicts still present in cleaned set")
        # If no conflicts after this iteration and all objects return
        return cleaned_set
    else:
        # Recursive step
        runners_up = multidimensional_deconfliction(association_set_reduced).associations

    for runner_up in runners_up:
        runner_up_remaining_time = runner_up.time_range
        for winner in cleaned_set:
            if conflicts(runner_up, winner):
                runner_up_remaining_time -= winner.time_range
        if runner_up_remaining_time and runner_up_remaining_time.duration.total_seconds() > 0:
            runner_up_copy = copy.copy(runner_up)
            runner_up_copy.time_range = runner_up_remaining_time
            cleaned_set.add(runner_up_copy)
    return cleaned_set


def conflicts(assoc1, assoc2):
    if hasattr(assoc1, 'time_range') and hasattr(assoc2, 'time_range') and \
            assoc1.objects.intersection(assoc2.objects) and \
            assoc1.time_range & assoc2.time_range and \
            assoc1 != assoc2:
        return True
    else:
        return False


def check_if_no_conflicts(association_set):
    for n, assoc1 in enumerate(association_set):
        for assoc2 in islice(association_set, n, None):
            if conflicts(assoc1, assoc2):
                return False
    return True
