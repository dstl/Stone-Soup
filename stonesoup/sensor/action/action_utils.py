import numpy as np

from ...types.angle import Angle


def contains_angle(min_, max_, item):
    """A utility function for angle-based action and action-generators to determine whether a
    given angle is within a minimum and maximum angle interval.

    Assumes that min_, max_ and item are within the interval (-180, 180) (degrees).
    However, min_, max_, item should be in radians.
    Contains the logic for the instance in which min_ > max_, ie. 180 degrees is within the
    interval."""
    if min_ < max_:
        if min_ <= item <= max_:
            return True
        else:
            return False

    else:
        if min_ <= item <= Angle(np.radians(180)) \
                or max_ <= item <= Angle(np.radians(-180)):
            return True
        else:
            return False
