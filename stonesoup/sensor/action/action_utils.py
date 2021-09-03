import numpy as np

from ...types.angle import Bearing


def contains_bearing(min_, max_, item, epsilon=1e-6):
    """A utility function for angle-based action and action-generators to determine whether a
    given angle is within a minimum and maximum angle interval.

    Casts min_, max_ and item as :class:`~.Bearing` types to keep within (-180, 180), then as
    floats of checking equivalence with tolerance `epsilon`.

    A tolerance of `epsilon` is used to account for floating point error in inequality checks.

    Contains logic for the instance in which min_ > max_."""

    min_, max_, item = float(Bearing(min_)), float(Bearing(max_)), float(Bearing(item))

    if min_ < max_:
        if min_ - epsilon <= item <= max_ + epsilon:
            return True
        else:
            return False
    else:
        if (min_ - epsilon <= item <= np.radians(180) + epsilon
                or np.radians(-180) - epsilon <= item <= max_ + epsilon):
            return True
        else:
            return False
