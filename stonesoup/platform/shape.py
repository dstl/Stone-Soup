from typing import Sequence, Union
import numpy as np

from stonesoup.base import Property, Base
from stonesoup.types.array import StateVectors


class Shape(Base):
    """
    A class for handling shapes as polygons which can be subsequently
    passed to :class:`~.Platform` when considering physical presence of
    the vehicle. This is also used when defining obstacles which require
    shape in order for simualted sensors to be affected.
    """

    shape_data: StateVectors = Property(
        default=None,
        doc="Coordinates defining the vertices of the obstacle relative"
        "to its centroid without any orientation. Defaults to `None`")

    simplices: Union[Sequence[int], np.ndarray] = Property(
        default=None,
        doc="A :class:`Sequence` or :class:`np.ndarray`, describing the connectivity "
            "of vertices specified in :attr:`shape_data`. Should be constructed such "
            "that element `i` is the index of a vertex that `i` is connected to. "
            "For example, simplices for a four sided obstacle may be `(1, 2, 3, 0)` "
            "for consecutively defined vertices. Default assumes that :attr:`shape_data` "
            "is provided such that consecutive vertices are connected, such as the "
            "example above.")

    shape_mapping: Sequence[int] = Property(
        default=(0, 1),
        doc="A mapping for :attr:`shape_data` dimensions to :math:`xy` Cartesian "
        "position. Default value is (0,1).")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # If simplices not defined, calculate based on sequential vertices assumption
        if self.simplices is None:
            self.simplices = np.roll(np.linspace(0,
                                                 self.shape_data.shape[1]-1,
                                                 self.shape_data.shape[1]),
                                     -1).astype(int)
